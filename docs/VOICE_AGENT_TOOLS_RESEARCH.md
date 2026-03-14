# Voice Agent com Acoes Reais — Pesquisa e Arquitetura

> Documento gerado em 2026-03-13.
> Pesquisa profunda sobre como construir agentes de voz hiper-realistas
> com tool calling (function calling) para acoes concretas.

---

## 1. O Problema Atual

O agente faz **promessas falsas** porque nao tem ferramentas para executar:

```
Usuario: "Gostaria de saber informacoes do meu cartao de credito."
Agente:  "Claro, vou verificar sua conta agora. Qual e o seu numero do cartao?"
Usuario: "1, 2, 3, 4."
Agente:  "Vou verificar as informacoes. Um momento, por favor."
         [NUNCA VERIFICA — NAO TEM FERRAMENTAS]
Usuario: esperando... para sempre
```

**Resultado:** Experiencia horrivel. O usuario confia, espera, e nada acontece.
O agente mente sem saber que esta mentindo.

---

## 2. Como Voice Agents Modernos Resolvem Isso

### 2.1 Arquitetura com Tool Calling

```
Usuario fala
    ↓
ASR → texto
    ↓
LLM (com tools definidos)
    ↓
    ├── Se LLM gera TEXTO → TTS → audio para usuario
    │
    └── Se LLM gera TOOL_CALL → executa ferramenta
            ↓
        Resultado da ferramenta
            ↓
        LLM recebe resultado → gera TEXTO → TTS → audio
```

**A diferenca fundamental:** O LLM decide SE vai chamar uma ferramenta
baseado no contexto da conversa. Se o usuario pede saldo, o LLM chama
`get_account_balance()` em vez de inventar uma resposta.

### 2.2 Tres Padroes de Arquitetura

#### Padrao A: Client-Side Execution (OpenAI Realtime API)

```
[Usuario fala] → [Realtime API] → [LLM gera function_call]
     ↓
[Client detecta function_call no response.output_item.done]
     ↓
[Client executa tool (HTTP, DB, etc.)]
     ↓
[Client envia conversation.item.create com function_call_output]
     ↓
[Client envia response.create]
     ↓
[LLM gera resposta falada com o resultado]
```

Usado por: OpenAI Realtime Console, openai-realtime-agents.

#### Padrao B: Webhook-Based (Vapi, Bland.ai, Retell.ai)

```
[Usuario fala] → [Plataforma ASR] → [LLM decide tool call]
     ↓
[Plataforma envia HTTP POST para seu servidor]
  {
    "type": "tool-calls",
    "toolCallList": [
      { "id": "abc123", "name": "check_balance", "arguments": {...} }
    ]
  }
     ↓
[Seu servidor executa e responde:]
  {
    "results": [
      { "toolCallId": "abc123", "result": "{\"balance\": 1250}" }
    ]
  }
     ↓
[Plataforma alimenta resultado no LLM → TTS → fala para usuario]
```

Usado por: Vapi, Bland.ai, Retell.ai.

#### Padrao C: Server-Side Pipeline (Pipecat, LiveKit Agents) — RECOMENDADO

```
[Usuario fala] → [ASR] → [LLM stream]
     ↓
[LLM stream emite tool_call em vez de texto]
     ↓
[Pipeline detecta, envia filler TTS: "Vou verificar..."]
     ↓
[Pipeline executa ferramenta server-side]
     ↓
[Resultado adicionado ao historico de mensagens]
     ↓
[Nova chamada LLM com resultado → texto → TTS → audio]
```

Usado por: Pipecat, LiveKit Agents, e a maioria dos voice agents open-source.

**Este e o padrao que devemos implementar.**

---

## 3. Protocolo OpenAI Realtime API — Function Calling

### 3.1 Definicao de Tools (session.update)

```json
{
  "type": "session.update",
  "session": {
    "tools": [
      {
        "type": "function",
        "name": "get_account_balance",
        "description": "Consulta o saldo da conta do cliente",
        "parameters": {
          "type": "object",
          "properties": {
            "account_id": {
              "type": "string",
              "description": "Numero da conta do cliente"
            }
          },
          "required": ["account_id"]
        }
      }
    ]
  }
}
```

### 3.2 Fluxo Completo de Eventos

```
1. LLM decide chamar ferramenta (em vez de gerar audio/texto)

2. response.function_call_arguments.delta
   → JSON dos argumentos chega incrementalmente

3. response.function_call_arguments.done
   → Argumentos completos: { call_id, name, arguments }

4. response.output_item.done
   → Item function_call finalizado

5. response.done
   → Response completa (LLM parou para esperar resultado)

6. [Seu codigo executa a ferramenta]

7. conversation.item.create  (client → server)
   {
     "type": "conversation.item.create",
     "item": {
       "type": "function_call_output",
       "call_id": "<mesmo call_id do passo 3>",
       "output": "{\"balance\": 1250.00, \"currency\": \"BRL\"}"
     }
   }

8. response.create  (client → server)
   → Dispara nova geracao do LLM

9. LLM fala o resultado
   → "Seu saldo atual e de R$1.250,00."
```

### 3.3 Insight Critico

Function calls sao **conversation items** — vivem no historico junto com
mensagens do usuario e do assistente. O LLM tem contexto completo de quais
tools foram chamados e quais resultados voltaram.

---

## 4. Function Calling em LLM Streaming

### 4.1 Como Tool Calls Aparecem no Stream

No formato OpenAI Chat Completions streaming, cada `chunk.choices[0].delta`
pode conter:

```python
# Resposta normal (texto):
delta.content = "Seu saldo"    # token de texto

# Tool call (fragmentos):
delta.tool_calls[0].index = 0
delta.tool_calls[0].id = "call_abc123"              # so no 1o chunk
delta.tool_calls[0].function.name = "check_balance"  # so no 1o chunk
delta.tool_calls[0].function.arguments = '{"ac'       # JSON parcial

# Chunks seguintes:
delta.tool_calls[0].function.arguments = 'count_'     # mais fragmentos JSON

# ... ate argumentos completos (finish_reason = "tool_calls")
```

### 4.2 Como a Pipeline se Comporta Durante Tool Execution

```
LLM stream emite tool_call events (nao texto)
    ↓
Nenhum texto entra no sentence buffer
    ↓
Nenhuma sentenca e enviada para TTS
    ↓
TTS naturalmente pausa (nao ha o que sintetizar)
    ↓
Tool executa (seu codigo roda a funcao)
    ↓
Resultado adicionado ao historico
    ↓
NOVA chamada LLM com resultado nas mensagens
    ↓
LLM stream agora emite texto
    ↓
TTS retoma naturalmente
```

### 4.3 O Loop de Execucao de Tools

```python
while True:
    # Stream LLM com tools
    response = stream_llm(messages, tools=tools)

    if response contem apenas texto:
        # Envia para TTS, fim
        yield text_to_tts
        break

    if response contem tool_calls:
        for tool_call in response.tool_calls:
            # Executa ferramenta
            result = await execute_tool(tool_call.name, tool_call.arguments)

            # Adiciona ao historico
            messages.append({
                "role": "assistant",
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        # Loop novamente — LLM recebe resultado e gera resposta
        continue
```

Este loop permite **multi-tool** (LLM chama varias ferramentas em sequencia)
e **tool chains** (resultado de uma ferramenta leva a outra chamada).

---

## 5. Filler Phrases e UX During Tool Execution

### 5.1 O Que Falar Enquanto Executa

| Situacao | Frase (PT-BR) |
|---|---|
| Iniciando consulta | "Vou verificar isso para voce." |
| Consulta rapida (<2s) | "Um momento." |
| Consulta demorada (>3s) | "Estou consultando o sistema, so mais um instante." |
| Falha na consulta | "Desculpe, nao consegui acessar essa informacao agora." |
| Timeout | "O sistema esta demorando para responder. Posso tentar novamente?" |

### 5.2 Implementacao de Fillers (Padrao Pipecat)

```python
@llm.event_handler("on_function_calls_started")
async def on_function_calls_started(service, function_calls):
    # Sintetiza filler ANTES de executar a tool
    await tts.queue_frame(TTSSpeakFrame("Vou verificar isso para voce."))
```

### 5.3 Anti-Patterns

| Errado | Certo |
|---|---|
| "Vou verificar" (sem ter ferramenta) | "Nao tenho acesso a essa informacao" |
| "Um momento" (e nunca volta) | "Um momento." (executa e volta com resultado) |
| "Estou consultando o banco de dados" | "Um instante, por favor." |
| Silencio total por >3s | Filler a cada 3s: "Ainda estou verificando..." |

### 5.4 Regra de Ouro do System Prompt

```
REGRAS ABSOLUTAS:
- Voce SO pode oferecer acoes para as quais tem ferramentas.
- NUNCA diga "vou verificar", "vou consultar", "vou checar" a menos
  que voce tenha uma ferramenta especifica para isso.
- Se o usuario pedir algo que voce NAO tem ferramenta para fazer,
  diga: "No momento nao consigo fazer isso por aqui, mas posso
  te transferir para um atendente."
- Quando voce TEM a ferramenta, USE-A imediatamente. Nao descreva
  o que vai fazer — faca.
```

---

## 6. Tools Tipicas para Customer Service por Voz

### 6.1 Conjunto Minimo Viavel

```json
[
  {
    "name": "lookup_customer",
    "description": "Busca cliente pelo numero de telefone ou CPF",
    "parameters": {
      "properties": {
        "phone": { "type": "string", "description": "Telefone com DDD" },
        "cpf": { "type": "string", "description": "CPF do cliente" }
      }
    }
  },
  {
    "name": "get_account_balance",
    "description": "Consulta saldo atual da conta",
    "parameters": {
      "properties": {
        "account_id": { "type": "string" }
      },
      "required": ["account_id"]
    }
  },
  {
    "name": "get_recent_transactions",
    "description": "Lista ultimas transacoes da conta",
    "parameters": {
      "properties": {
        "account_id": { "type": "string" },
        "limit": { "type": "integer", "default": 5 }
      },
      "required": ["account_id"]
    }
  },
  {
    "name": "get_card_info",
    "description": "Consulta informacoes do cartao de credito",
    "parameters": {
      "properties": {
        "card_number": { "type": "string" }
      },
      "required": ["card_number"]
    }
  },
  {
    "name": "create_support_ticket",
    "description": "Cria ticket de suporte para acompanhamento",
    "parameters": {
      "properties": {
        "category": { "type": "string", "enum": ["billing", "technical", "general"] },
        "description": { "type": "string" },
        "priority": { "type": "string", "enum": ["low", "medium", "high"] }
      },
      "required": ["category", "description"]
    }
  },
  {
    "name": "transfer_to_human",
    "description": "Transfere ligacao para atendente humano",
    "parameters": {
      "properties": {
        "department": { "type": "string", "enum": ["billing", "technical", "retention"] },
        "reason": { "type": "string" }
      },
      "required": ["department"]
    }
  }
]
```

### 6.2 Tools Avancadas

| Tool | Descricao | Complexidade |
|---|---|---|
| schedule_callback | Agenda retorno de ligacao | Media |
| send_sms | Envia SMS com informacoes | Baixa |
| verify_identity | Verifica identidade via perguntas | Alta |
| process_payment | Processa pagamento | Alta (requer confirmacao) |
| cancel_service | Cancela servico | Alta (requer confirmacao + retention flow) |

### 6.3 Tools que Requerem Confirmacao

Algumas acoes sao destrutivas e precisam de confirmacao explicita:

```python
{
    "name": "cancel_subscription",
    "description": "Cancela assinatura do cliente. IMPORTANTE: sempre confirme com o cliente antes de executar.",
    "parameters": {
        "account_id": { "type": "string" },
        "confirmation": {
            "type": "boolean",
            "description": "True apenas se o cliente confirmou explicitamente que quer cancelar"
        }
    },
    "required": ["account_id", "confirmation"]
}
```

---

## 7. Estado Atual do Codebase

### 7.1 O Que JA Existe

O macaw-voice-agent ja tem infraestrutura significativa para tool calling:

| Componente | Arquivo | Status |
|---|---|---|
| LLMStreamEvent com tool_call types | `providers/llm.py` | Implementado |
| Parsing de tool calls em streaming | `providers/llm_openai.py` | Implementado |
| `generate_stream_with_tools()` | `providers/llm_openai.py` | Implementado |
| `items_to_messages()` com function_call items | `pipeline/conversation.py` | Implementado |
| `_generate_response_with_tools()` | `server/session.py` | Implementado |
| Eventos protocol function_call | `protocol/events.py` | Implementado |

### 7.2 O Que FALTA (Gap Critico)

**O sistema emite tool calls para o CLIENT e espera o client executar.**
Isso e o protocolo OpenAI Realtime API (client-side execution).

Para um voice agent autonomo, precisamos de **server-side execution**:

| Faltando | Descricao |
|---|---|
| **Tool Registry** | Registro de ferramentas com handlers async no servidor |
| **Tool Execution Loop** | Quando LLM emite tool_call, executa server-side e re-chama LLM |
| **Filler Mechanism** | Sintetiza "Vou verificar..." via TTS enquanto tool executa |
| **Tool Timeout/Error** | Timeout de 5-10s por tool, error handling graceful |
| **Tool Definitions Config** | Definicoes de tools configuráveis (JSON/env) |
| **Mock Tool Handlers** | Handlers mock para demo/teste sem backend real |
| **System Prompt Update** | Prompt que referencia tools disponiveis e regras de uso |

### 7.3 Fluxo Ideal Pos-Implementacao

```
Usuario: "Qual e o saldo do meu cartao?"

LLM decide: tool_call → get_card_info(card_number="1234")

Pipeline:
  1. Detecta tool_call no stream
  2. TTS sintetiza: "Vou verificar isso para voce."
  3. Executa get_card_info("1234") → { "balance": 1250.00 }
  4. Adiciona resultado ao historico
  5. Nova chamada LLM com resultado
  6. LLM gera: "O saldo do seu cartao e de R$1.250,00."
  7. TTS sintetiza resposta final

Usuario ouve:
  "Vou verificar isso para voce. [pausa 1-2s] O saldo do seu cartao
   e de mil duzentos e cinquenta reais."
```

---

## 8. Implementacao Recomendada

### 8.1 Fase 1 — Tool Registry + Execution Loop

Criar um sistema de registro de tools no servidor:

```python
# Conceitual — nao e codigo final
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, name, handler, schema):
        self._tools[name] = ToolDef(name, handler, schema)

    async def execute(self, name, arguments) -> str:
        tool = self._tools[name]
        result = await asyncio.wait_for(
            tool.handler(**arguments),
            timeout=10.0
        )
        return json.dumps(result)

    def get_schemas(self) -> list[dict]:
        return [t.schema for t in self._tools.values()]
```

### 8.2 Fase 2 — Modificar Pipeline de Resposta

O `_generate_response_with_tools()` em `session.py` precisa:

1. Detectar `tool_call_end` no stream
2. Sintetizar filler via TTS
3. Executar tool server-side
4. Adicionar resultado ao historico
5. Re-chamar LLM
6. Continuar pipeline com texto do LLM

### 8.3 Fase 3 — Filler TTS

Quando tool_call e detectado:

```python
# Sintetizar filler imediatamente
filler = "Vou verificar isso para voce."
filler_audio = await tts.synthesize(filler)
await send_audio_to_client(filler_audio)

# Executar tool
result = await tool_registry.execute(tool_name, tool_args)
```

### 8.4 Fase 4 — System Prompt Atualizado

```
Voce e a Sara, atendente por telefone. Fale em portugues brasileiro.

FERRAMENTAS DISPONIVEIS:
- get_card_info: consulta informacoes do cartao
- get_account_balance: consulta saldo da conta
- transfer_to_human: transfere para atendente humano

REGRAS:
- Responda em NO MAXIMO 1 frase curta.
- Use as ferramentas quando o cliente pedir informacoes de conta/cartao.
- NUNCA prometa fazer algo que nao esta nas ferramentas acima.
- Se o cliente pedir algo fora das ferramentas, diga que vai transferir
  para um atendente.
- Quando usar uma ferramenta, nao diga "vou verificar" — a ferramenta
  ja sera executada automaticamente.
```

### 8.5 Fase 5 — Mock Handlers para Demo

```python
async def mock_get_card_info(card_number: str) -> dict:
    return {
        "card_number": f"****{card_number[-4:]}",
        "balance": 1250.00,
        "limit": 5000.00,
        "due_date": "2026-04-15",
        "minimum_payment": 125.00,
    }

async def mock_get_account_balance(account_id: str) -> dict:
    return {
        "account_id": account_id,
        "balance": 3450.75,
        "available": 3200.00,
    }

async def mock_transfer_to_human(department: str, reason: str = "") -> dict:
    return {
        "status": "transferred",
        "department": department,
        "queue_position": 3,
        "estimated_wait": "2 minutos",
    }
```

---

## 9. Comparacao com Plataformas Comerciais

| Feature | OpenAI Realtime | Vapi | Bland.ai | Nosso (atual) | Nosso (futuro) |
|---|---|---|---|---|---|
| ASR | Integrado | Integrado | Integrado | Whisper (remote) | Whisper (remote) |
| LLM | GPT-4o | Qualquer | Qualquer | Qwen2.5-7B | Qwen2.5-7B |
| TTS | Integrado | Integrado | Integrado | Kokoro (remote) | Kokoro (remote) |
| Tool Calling | Client-side | Webhook | Webhook | Nao tem | Server-side |
| Filler phrases | Nao | Config por tool | Config | Nao | Automatico |
| Tool timeout | N/A (client) | Configuravel | Configuravel | N/A | 10s default |
| Multi-tool | Sim | Sim | Sim | N/A | Sim (loop) |
| Custo | $$$$ | $$$ | $$$ | $ (self-hosted) | $ (self-hosted) |

---

## 10. Prioridade de Implementacao

| Prioridade | Item | Impacto | Esforco |
|---|---|---|---|
| P0 | Atualizar system prompt para NAO prometer | Critico | Baixo |
| P1 | Tool Registry + schemas | Alto | Medio |
| P1 | Tool Execution Loop no session.py | Alto | Medio |
| P2 | Filler TTS durante execucao | Medio | Baixo |
| P2 | Mock handlers para demo | Medio | Baixo |
| P3 | Tool definitions via config | Baixo | Baixo |
| P3 | Handlers reais (API banking) | Alto | Alto |
| P3 | Multi-agent handoff | Medio | Alto |

**Acao imediata (P0):** Mesmo sem implementar tools, o system prompt
DEVE ser atualizado para nao fazer promessas falsas. Isso resolve
o problema mais grave com zero codigo.

---

## 11. Recursos e Referencias

### Frameworks Open-Source
- **Pipecat** — Pipeline-based voice agent framework (Daily.co)
- **LiveKit Agents** — WebRTC voice agent SDK
- **openai-realtime-agents** — Multi-agent com tool calling

### Plataformas Comerciais (referencia)
- **Vapi.ai** — Voice agent platform com webhook tools
- **Bland.ai** — Enterprise voice agents
- **Retell.ai** — Voice agent platform

### Documentacao
- OpenAI Realtime API — Function Calling
- OpenAI Chat Completions — Tool Use / Parallel Function Calling
- vLLM — Tool calling support (Qwen2.5 suporta)
