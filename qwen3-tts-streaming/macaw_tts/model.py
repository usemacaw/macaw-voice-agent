"""MacawTTS — High-level wrapper for streaming Qwen3-TTS.

Loads model, manages CUDA graphs, exposes simple streaming API.
Zero external dependencies beyond qwen-tts and torch.

Usage:
    tts = MacawTTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    for chunk, sr, meta in tts.stream("Olá, como posso ajudar?", language="Portuguese"):
        play_audio(chunk, sr)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch

from macaw_tts.audio import CODEC_SAMPLE_RATE
from macaw_tts.crossfade import HannCrossfader
from macaw_tts.decoder import StreamingDecoder
from macaw_tts.predictor_graph import PredictorCUDAGraph
from macaw_tts.streaming import ChunkMetadata, streaming_generate
from macaw_tts.talker_graph import TalkerCUDAGraph

logger = logging.getLogger("macaw-tts")


class MacawTTS:
    """High-level streaming TTS using Qwen3-TTS with CUDA graphs.

    ⚠️ from_pretrained() REQUIRES GPU.
    """

    def __init__(
        self,
        base_model,
        predictor_graph: PredictorCUDAGraph,
        talker_graph: TalkerCUDAGraph,
        decoder: StreamingDecoder,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._model = base_model
        self._predictor_graph = predictor_graph
        self._talker_graph = talker_graph
        self._decoder = decoder
        self._device = device
        self._dtype = dtype
        self._warmed_up = False

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 2048,
        decode_window: int = 40,
        context_frames: int = 25,
        compile_decoder: bool = True,
    ) -> "MacawTTS":
        """Load Qwen3-TTS and prepare CUDA graphs.

        🟢 GPU REQUIRED.

        Uses SDPA attention (not flash-attn) because CUDA graphs are
        incompatible with flash_attention_2. Validated on RTX 4090:
        SDPA+graphs gives TTFA=64ms, RTF=3.49x.
        """
        attn_implementation = "sdpa"
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        # IMPORTANT: flash_attention_2 is INCOMPATIBLE with CUDA graphs.
        # Tested on RTX 4090 (2026-03-20):
        #   - flash-attn + dynamic cache: RTF 0.54x (slower than real-time)
        #   - flash-attn + torch.compile: crashes (InternalTorchDynamoError)
        #   - SDPA + CUDA graphs: RTF 3.49x, TTFA 64ms (BEST)
        # Force SDPA regardless of flash-attn availability.
        if attn_implementation != "sdpa":
            logger.warning(
                f"attn_implementation='{attn_implementation}' requested but "
                "flash_attention_2 is incompatible with CUDA graphs. Forcing 'sdpa'."
            )
            attn_implementation = "sdpa"

        logger.info(f"Loading Qwen3-TTS: {model_name} on {device} (attn={attn_implementation})")
        base_model = Qwen3TTSModel.from_pretrained(
            model_name, device_map=device, dtype=dtype,
            attn_implementation=attn_implementation,
        )

        talker = base_model.model.talker
        predictor = talker.code_predictor
        talker_config = base_model.model.config.talker_config
        pred_config = predictor.model.config

        predictor_graph = PredictorCUDAGraph(
            predictor, pred_config, talker_config.hidden_size,
            device=device, dtype=dtype,
        )
        talker_graph_obj = TalkerCUDAGraph(
            talker.model, talker_config,
            device=device, dtype=dtype, max_seq_len=max_seq_len,
        )

        speech_tokenizer = base_model.model.speech_tokenizer
        decoder = StreamingDecoder(
            speech_tokenizer, decode_window=decode_window,
            context_frames=context_frames,
        )
        if compile_decoder:
            decoder.compile(mode="reduce-overhead")

        logger.info("SDPA + CUDA graphs (captured on first stream())")
        return cls(
            base_model, predictor_graph, talker_graph_obj, decoder,
            device=device, dtype=dtype,
        )

    def _warmup(self, prefill_len: int) -> None:
        """Capture CUDA graphs. 🟢 GPU REQUIRED."""
        logger.info("Capturing CUDA graphs...")
        self._predictor_graph.capture(num_warmup=3)
        self._talker_graph.capture(prefill_len=prefill_len, num_warmup=3)
        self._warmed_up = True
        logger.info("CUDA graphs ready!")

    # =========================================================================
    # Input building — adapted from upstream Qwen3-TTS generate()
    # No external dependencies. Builds talker_input_embeds directly.
    # =========================================================================

    def _build_talker_inputs(
        self,
        m,
        input_ids: List[torch.Tensor],
        ref_ids: Optional[List[Optional[torch.Tensor]]],
        voice_clone_prompt: Optional[Dict[str, Any]],
        languages: List[str],
        speakers: Optional[List[Optional[str]]],
        instruct_ids: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build talker input embeddings from text/speaker/language.

        Mirrors upstream Qwen3TTSForConditionalGeneration.generate() logic.
        Returns (talker_input_embeds, attention_mask, trailing_text_hiddens, tts_pad_embed).
        """
        talker_input_embeds = [[] for _ in range(len(input_ids))]

        # Voice clone speaker prompt
        voice_clone_spk_embeds = None
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = m.generate_speaker_prompt(voice_clone_prompt)

        # Instruct text
        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(
                        m.talker.text_projection(m.talker.get_text_embeddings()(instruct_id))
                    )

        if speakers is None:
            speakers = [None] * len(input_ids)

        trailing_text_hiddens = []
        tts_pad_embed = None

        for index, (input_id, language, speaker) in enumerate(
            zip(input_ids, languages, speakers)
        ):
            # Speaker embedding
            if voice_clone_spk_embeds is None:
                if speaker in ("", None):
                    speaker_embed = None
                else:
                    spk_key = speaker.lower()
                    if spk_key not in m.config.talker_config.spk_id:
                        raise ValueError(f"Speaker '{speaker}' not found")
                    spk_id = m.config.talker_config.spk_id[spk_key]
                    speaker_embed = m.talker.get_input_embeddings()(
                        torch.tensor(spk_id, device=m.talker.device, dtype=input_id.dtype)
                    )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            # Language ID
            lang_lower = language.lower() if language else "auto"
            if lang_lower == "auto":
                language_id = None
            else:
                if lang_lower not in m.config.talker_config.codec_language_id:
                    raise ValueError(f"Language '{language}' not supported")
                language_id = m.config.talker_config.codec_language_id[lang_lower]

            # Dialect handling
            if (
                lang_lower in ["chinese", "auto"]
                and speaker not in ("", None)
                and hasattr(m.config.talker_config, "spk_is_dialect")
                and speaker.lower() in m.config.talker_config.spk_is_dialect
            ):
                dialect = m.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = m.config.talker_config.codec_language_id[dialect]

            # BOS/EOS/PAD embeddings
            tts_bos_embed, tts_eos_embed, tts_pad_embed = m.talker.text_projection(
                m.talker.get_text_embeddings()(
                    torch.tensor(
                        [[m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id]],
                        device=m.talker.device, dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)

            # Codec prefill tokens
            if language_id is None:
                codec_prefill = [[
                    m.config.talker_config.codec_nothink_id,
                    m.config.talker_config.codec_think_bos_id,
                    m.config.talker_config.codec_think_eos_id,
                ]]
            else:
                codec_prefill = [[
                    m.config.talker_config.codec_think_id,
                    m.config.talker_config.codec_think_bos_id,
                    language_id,
                    m.config.talker_config.codec_think_eos_id,
                ]]

            codec_emb_0 = m.talker.get_input_embeddings()(
                torch.tensor(codec_prefill, device=m.talker.device, dtype=input_id.dtype)
            )
            codec_emb_1 = m.talker.get_input_embeddings()(
                torch.tensor(
                    [[m.config.talker_config.codec_pad_id, m.config.talker_config.codec_bos_id]],
                    device=m.talker.device, dtype=input_id.dtype,
                )
            )

            if speaker_embed is None:
                codec_emb = torch.cat([codec_emb_0, codec_emb_1], dim=1)
            else:
                codec_emb = torch.cat([codec_emb_0, speaker_embed.view(1, 1, -1), codec_emb_1], dim=1)

            # Role embedding (first 3 tokens of input: <|im_start|>assistant\n)
            role_embed = m.talker.text_projection(
                m.talker.get_text_embeddings()(input_id[:, :3])
            )

            # Dual-track fusion: text_pad + codec
            talker_embed = torch.cat(
                (
                    tts_pad_embed.expand(-1, codec_emb.shape[1] - 2, -1),
                    tts_bos_embed,
                ),
                dim=1,
            ) + codec_emb[:, :-1]

            talker_embed = torch.cat((role_embed, talker_embed), dim=1)

            # ICL mode (voice cloning with reference audio)
            if (
                voice_clone_prompt is not None
                and voice_clone_prompt.get("ref_code") is not None
                and voice_clone_prompt["icl_mode"][index]
            ):
                icl_embed, trailing_text_hidden = m.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(m.talker.device).clone(),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=False,
                )
                talker_embed = torch.cat([talker_embed, icl_embed], dim=1)
            else:
                # Standard mode: add first text token + codec_bos
                talker_embed = torch.cat(
                    [
                        talker_embed,
                        m.talker.text_projection(
                            m.talker.get_text_embeddings()(input_id[:, 3:4])
                        ) + codec_emb[:, -1:],
                    ],
                    dim=1,
                )
                # Trailing text: remaining text tokens + EOS
                trailing_text_hidden = torch.cat(
                    (
                        m.talker.text_projection(
                            m.talker.get_text_embeddings()(input_id[:, 4:-5])
                        ),
                        tts_eos_embed,
                    ),
                    dim=1,
                )

            talker_input_embeds[index].append(talker_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        # Combine and pad
        for index, parts in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat(
                [p for p in parts if p is not None], dim=1
            )

        # Build attention mask with left-padding
        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed, batch_first=True, padding_value=0.0,
        )
        tie = padded_reversed.flip(dims=[1])

        batch_size, max_len = tie.shape[0], tie.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        tam = (indices >= num_pads.unsqueeze(1)).long().to(tie.device)

        # Pad trailing text hiddens
        pad_vec = tts_pad_embed.squeeze()
        seqs = [t.squeeze(0) for t in trailing_text_hiddens]
        orig_lens = [s.shape[0] for s in seqs]
        tth = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
        arange = torch.arange(max(orig_lens), device=tth.device).expand(len(orig_lens), -1)
        lens = torch.tensor(orig_lens, device=tth.device).unsqueeze(1)
        padding_mask = arange >= lens
        tth[padding_mask] = pad_vec

        return tie, tam, tth, tts_pad_embed

    # =========================================================================
    # Public API
    # =========================================================================

    def stream(
        self,
        text: str,
        language: str = "Portuguese",
        speaker: Optional[str] = None,
        *,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_frames: int = 2048,
        emit_every_phase1: int = 1,
        emit_every_phase2: int = 4,
        phase1_frames: int = 1,
        overlap_samples: int = 512,
    ) -> Generator[tuple[np.ndarray, int, ChunkMetadata], None, None]:
        """Stream audio from text. 🟢 GPU REQUIRED.

        Phase 1: emit first frame immediately (~88ms TTFA).
        Phase 2: emit every 4 frames (~333ms per chunk).

        Yields (audio_chunk_float32, sample_rate, metadata).
        """
        m = self._model.model
        input_texts = [self._model._build_assistant_text(text)]
        input_ids = self._model._tokenize_texts(input_texts)

        tie, tam, tth, tpe = self._build_talker_inputs(
            m, input_ids, ref_ids=[None],
            voice_clone_prompt=None,
            languages=[language] if language else ["Auto"],
            speakers=[speaker],
        )

        if not self._warmed_up:
            self._warmup(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None
        crossfader = HannCrossfader(overlap_samples=overlap_samples)

        yield from streaming_generate(
            talker, tie, tam, tth, tpe, config,
            self._predictor_graph, self._talker_graph,
            self._decoder, crossfader,
            max_new_tokens=max_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=True, repetition_penalty=repetition_penalty,
            emit_every_phase1=emit_every_phase1,
            emit_every_phase2=emit_every_phase2,
            phase1_frames=phase1_frames,
        )

    def stream_voice_clone(
        self,
        text: str,
        language: str = "Portuguese",
        ref_audio: str = "",
        ref_text: str = "",
        *,
        voice_clone_prompt: Optional[Dict[str, Any]] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_frames: int = 2048,
        emit_every_phase1: int = 1,
        emit_every_phase2: int = 4,
        phase1_frames: int = 1,
        overlap_samples: int = 512,
    ) -> Generator[tuple[np.ndarray, int, ChunkMetadata], None, None]:
        """Stream audio with voice cloning. 🟢 GPU REQUIRED."""
        m = self._model.model

        # Create voice clone prompt if not provided
        if voice_clone_prompt is None:
            prompt_items = self._model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text,
            )
            voice_clone_prompt = m._prompt_items_to_voice_clone_prompt(prompt_items)

        # Build ref_ids for ICL
        ref_ids = [None]
        if voice_clone_prompt.get("icl_mode", [False])[0] and ref_text:
            ref_texts = [self._model._build_assistant_text(ref_text)]
            ref_ids = self._model._tokenize_texts(ref_texts)

        input_texts = [m._build_assistant_text(text)]
        input_ids = m._tokenize_texts(input_texts)

        tie, tam, tth, tpe = self._build_talker_inputs(
            m, input_ids, ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt,
            languages=[language] if language else ["Auto"],
            speakers=[None],
        )

        if not self._warmed_up:
            self._warmup(tie.shape[1])

        # Extract ref_codes for decoder context
        ref_codes = None
        if (
            voice_clone_prompt.get("ref_code")
            and voice_clone_prompt["ref_code"][0] is not None
            and voice_clone_prompt.get("icl_mode", [False])[0]
        ):
            ref_codes = voice_clone_prompt["ref_code"][0].to(self._device)

        if not self._warmed_up:
            self._warmup(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None
        crossfader = HannCrossfader(overlap_samples=overlap_samples)

        yield from streaming_generate(
            talker, tie, tam, tth, tpe, config,
            self._predictor_graph, self._talker_graph,
            self._decoder, crossfader,
            max_new_tokens=max_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=True, repetition_penalty=repetition_penalty,
            emit_every_phase1=emit_every_phase1,
            emit_every_phase2=emit_every_phase2,
            phase1_frames=phase1_frames,
            ref_codes=ref_codes,
        )

    @property
    def sample_rate(self) -> int:
        return CODEC_SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device
