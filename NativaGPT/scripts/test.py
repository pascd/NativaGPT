import requests
import json
import os
import uuid
import sys
from typing import Dict, List

# --- CRIE ESTES FICHEIROS PARA TESTAR (ou altere os caminhos) ---
# ⚠️ GARANTA QUE ESTES CAMINHOS SÃO VÁLIDOS NO SEU SISTEMA!
IMAGE_PATHS = [
    os.path.join(os.path.dirname(__file__), "/home/pedro/Pictures/20mp_camera.png"),
    os.path.join(os.path.dirname(__file__), "/home/pedro/Pictures/robot_without_forces.png")
]
# ------------------------------------------------------------------


# === CONFIGURAÇÃO DA API ===
ENDPOINT = "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream"
API_KEY = "sk-usr-a7o2l7lsb84hmw4ybckltym6saz3fs1jflv"
CHANNEL_ID = "cmh92y20g0o48gt01sy1l5k9g"
THREAD_ID = str(uuid.uuid4())

# Campos de texto (Payload)
PAYLOAD_DATA = {
    "channel_id": CHANNEL_ID,
    "thread_id": THREAD_ID,
    "user_info": "{}",
    "message": "Analise o que é mostrado nestas duas imagens e diga-me o que têm em comum."
}

# Novo Timeout (aumentado para 60 segundos)
REQUEST_TIMEOUT = 60


def send_multi_image_request(endpoint: str, api_key: str, data: Dict[str, str], image_paths: List[str]):
    """Envia o pedido POST com dados e uma lista de ficheiros de imagem e processa o stream."""

    files_to_close = []

    try:
        files = []

        for path in image_paths:
            if not os.path.exists(path):
                print(f"❌ Ficheiro não encontrado: {path}. A ignorar.", file=sys.stderr)
                continue

            try:
                file_handle = open(path, 'rb')
                files_to_close.append(file_handle)
                mime_type = 'image/png' if path.lower().endswith('.png') else 'image/jpeg'

                files.append(
                    (
                        'image',
                        (os.path.basename(path), file_handle, mime_type)
                    )
                )
                print(f"✅ Preparado para enviar: {os.path.basename(path)}")

            except Exception as e:
                print(f"❌ Erro ao abrir ficheiro {path}: {e}", file=sys.stderr)

        if not files:
            print("⚠️ Nenhuma imagem válida para enviar.", file=sys.stderr)
            return

        headers = {'x-api-key': api_key}

        # 🔑 CORREÇÃO CRÍTICA: Adicionar stream=True e aumentar o timeout
        response = requests.post(
            endpoint,
            headers=headers,
            data=data,
            files=files,
            stream=True,         # <-- Permite ler a resposta à medida que chega
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        print("\n--- Resposta da LLM (Stream) ---")
        full_response = ""

        # Iterar sobre o conteúdo para processar o stream (NDJSON)
        # O iter_lines é ótimo para NDJSON
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    # Tentar descodificar a linha como um objeto JSON (NDJSON)
                    event = json.loads(line)
                    content = event.get("content", "")

                    if event.get("type") == "token" and content:
                        print(content, end="", flush=True)
                        full_response += content

                    # Se receber a mensagem final (ou 'done'), parar
                    elif event.get("type") == "message" and event.get("content", {}).get("content"):
                        final_content = event["content"]["content"]
                        print(final_content, end="", flush=True)
                        full_response += final_content
                        break

                except json.JSONDecodeError:
                    # Se não for JSON (ex: um erro de texto simples), imprima
                    print(line, end="", flush=True)
                    full_response += line

        print("\n------------------------------")
        return {"success": True, "full_response": full_response}


    except requests.exceptions.Timeout as e:
        print(f"\n❌ ERRO: Timeout após {REQUEST_TIMEOUT} segundos. A API do LLM pode estar lenta ou a(s) imagem(ns) ser(em) muito grande(s).", file=sys.stderr)
        return {"success": False, "error": str(e)}

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Erro na requisição: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
             print(f"Detalhes do erro do servidor ({e.response.status_code}): {e.response.text}", file=sys.stderr)
        return {"success": False, "error": str(e)}

    finally:
        # Fechar todos os handles dos ficheiros
        for handle in files_to_close:
            handle.close()


# === EXECUÇÃO ===
if __name__ == "__main__":
    print("--- Teste de Upload de Múltiplas Imagens (Corrigido) ---")

    send_multi_image_request(ENDPOINT, API_KEY, PAYLOAD_DATA, IMAGE_PATHS)