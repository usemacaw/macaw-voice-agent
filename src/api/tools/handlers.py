"""
Mock tool handlers for demo and testing.

These simulate real backend responses with realistic data.
Replace with real API calls for production use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.registry import ToolRegistry

logger = logging.getLogger("open-voice-api.tools.mock")


# ---------------------------------------------------------------------------
# Mock Handlers
# ---------------------------------------------------------------------------


async def mock_lookup_customer(phone: str = "", cpf: str = "") -> dict:
    """Busca cliente pelo telefone ou CPF."""
    if not phone and not cpf:
        return {"error": "Informe o telefone ou CPF do cliente."}

    return {
        "customer_id": "CLI-78542",
        "name": "Maria Silva",
        "phone": phone or "(11) 98765-4321",
        "cpf": cpf or "***.***.***-90",
        "account_id": "ACC-12345",
        "status": "active",
    }


async def mock_get_account_balance(account_id: str) -> dict:
    """Consulta saldo da conta."""
    return {
        "account_id": account_id,
        "balance": 3450.75,
        "available": 3200.00,
        "currency": "BRL",
        "updated_at": "2026-03-13T14:30:00-03:00",
    }


async def mock_get_card_info(card_number: str) -> dict:
    """Consulta informacoes do cartao de credito."""
    last4 = card_number[-4:] if len(card_number) >= 4 else card_number
    return {
        "card_number": f"**** **** **** {last4}",
        "card_type": "Visa Platinum",
        "balance": 1250.00,
        "limit": 5000.00,
        "available_limit": 3750.00,
        "due_date": "2026-04-15",
        "minimum_payment": 125.00,
        "currency": "BRL",
    }


async def mock_get_recent_transactions(
    account_id: str, limit: int = 5
) -> dict:
    """Lista ultimas transacoes da conta."""
    transactions = [
        {
            "date": "2026-03-13",
            "description": "Supermercado Extra",
            "amount": -187.50,
            "type": "debit",
        },
        {
            "date": "2026-03-12",
            "description": "PIX Recebido - Joao Santos",
            "amount": 500.00,
            "type": "credit",
        },
        {
            "date": "2026-03-11",
            "description": "Netflix",
            "amount": -55.90,
            "type": "debit",
        },
        {
            "date": "2026-03-10",
            "description": "Farmacia Drogasil",
            "amount": -42.30,
            "type": "debit",
        },
        {
            "date": "2026-03-09",
            "description": "Transferencia TED",
            "amount": -1200.00,
            "type": "debit",
        },
    ]
    return {
        "account_id": account_id,
        "transactions": transactions[:limit],
        "total_shown": min(limit, len(transactions)),
    }


async def mock_create_support_ticket(
    category: str, description: str, priority: str = "medium"
) -> dict:
    """Cria ticket de suporte."""
    return {
        "ticket_id": "TKT-2026031300042",
        "category": category,
        "priority": priority,
        "status": "open",
        "message": f"Ticket criado com sucesso. Numero do protocolo: TKT-2026031300042.",
    }


async def mock_transfer_to_human(
    department: str, reason: str = ""
) -> dict:
    """Transfere para atendente humano."""
    wait_times = {
        "billing": "2 minutos",
        "technical": "5 minutos",
        "retention": "1 minuto",
    }
    return {
        "status": "transferring",
        "department": department,
        "queue_position": 3,
        "estimated_wait": wait_times.get(department, "3 minutos"),
        "message": f"Transferindo para o setor de {department}.",
    }


# ---------------------------------------------------------------------------
# Tool Schemas (OpenAI function format)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Busca informacoes do cliente pelo numero de telefone ou CPF. Use quando o cliente se identificar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": "Telefone do cliente com DDD",
                    },
                    "cpf": {
                        "type": "string",
                        "description": "CPF do cliente",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_balance",
            "description": "Consulta o saldo atual da conta do cliente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "ID da conta do cliente",
                    },
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_card_info",
            "description": "Consulta informacoes do cartao de credito: saldo, limite, vencimento.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_number": {
                        "type": "string",
                        "description": "Numero do cartao de credito",
                    },
                },
                "required": ["card_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_transactions",
            "description": "Lista as ultimas transacoes da conta do cliente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "ID da conta do cliente",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Quantidade de transacoes (padrao: 5)",
                    },
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Cria um ticket de suporte para acompanhamento posterior.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["billing", "technical", "general"],
                        "description": "Categoria do ticket",
                    },
                    "description": {
                        "type": "string",
                        "description": "Descricao do problema",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Prioridade (padrao: medium)",
                    },
                },
                "required": ["category", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Transfere a ligacao para um atendente humano. Use quando o cliente pedir para falar com uma pessoa ou quando nao conseguir resolver a solicitacao.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "enum": ["billing", "technical", "retention"],
                        "description": "Setor para transferencia",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Motivo da transferencia",
                    },
                },
                "required": ["department"],
            },
        },
    },
]

_HANDLER_MAP: dict[str, tuple] = {
    "lookup_customer": (mock_lookup_customer, "Vou buscar seus dados."),
    "get_account_balance": (mock_get_account_balance, "Vou consultar seu saldo."),
    "get_card_info": (mock_get_card_info, "Vou verificar as informacoes do seu cartao."),
    "get_recent_transactions": (
        mock_get_recent_transactions,
        "Vou consultar suas ultimas transacoes.",
    ),
    "create_support_ticket": (mock_create_support_ticket, "Vou criar o ticket para voce."),
    "transfer_to_human": (mock_transfer_to_human, "Vou transferir voce agora."),
}


def register_mock_handlers(registry: ToolRegistry) -> None:
    """Register all mock handlers with their schemas."""
    schema_by_name = {s["function"]["name"]: s for s in _TOOL_SCHEMAS}

    for name, (handler, filler) in _HANDLER_MAP.items():
        schema = schema_by_name.get(name)
        if schema is None:
            logger.warning(f"No schema found for mock handler '{name}' — skipping")
            continue
        registry.register(name, handler=handler, schema=schema, filler_phrase=filler)

    logger.info(f"Registered {len(_HANDLER_MAP)} mock tool handlers")
