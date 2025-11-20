import requests
import json
import os
import time
from dotenv import load_dotenv
import uuid

load_dotenv()

ENDPOINT = "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream"
API_KEY = os.getenv('API_KEY')

headers = {'x-api-key': API_KEY}

def send_message(prompt):
    """Send message with detailed timing."""

    print("\n" + "="*80)
    print(f"📝 Prompt: {prompt}")
    print("="*80)

    # Gere um ID novo a cada execução ou quando quiser limpar a memória
    NEW_THREAD_ID = str(uuid.uuid4())

    data = {
        "channel_id": "cmh92y20g0o48gt01sy1l5k9g",
        "thread_id": NEW_THREAD_ID, # <--- MUDE ISTO AQUI
        "user_info": "{}",
        "message": prompt,
    }

    # Timing checkpoints
    timings = {}

    try:
        # 1. Start timing
        start_time = time.time()
        print(f"⏰ [0.00s] Starting request...")

        # 2. Send request
        response = requests.post(
            ENDPOINT,
            headers=headers,
            data=data,
            stream=True,
            timeout=120
        )

        request_sent_time = time.time()
        timings['request_sent'] = request_sent_time - start_time
        print(f"⏰ [{timings['request_sent']:.2f}s] Request sent, waiting for response...")

        response.raise_for_status()

        # 3. First byte received
        first_byte_time = None
        first_token_time = None
        last_token_time = None
        token_count = 0

        print(f"\n🤖 LLM: ", end='', flush=True)

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            # DEBUG: Imprimir a linha antes de tentar decodificar JSON
            print(f"\n[RAW LINE RECEIVED]: {line}") # <-- ADICIONE ISTO!

            # Mark first byte
            if first_byte_time is None:
                first_byte_time = time.time()
                timings['first_byte'] = first_byte_time - start_time
                print(f"\n⏰ [{timings['first_byte']:.2f}s] First response received")
                print(f"🤖 LLM: ", end='', flush=True)

            try:
                event = json.loads(line)
                event_type = event.get("type")

                if event_type == "start":
                    run_id = event.get("run_id")
                    print(f"\n⏰ [{time.time() - start_time:.2f}s] Stream started (ID: {run_id})")
                    print(f"🤖 LLM: ", end='', flush=True)

                elif event_type == "token":
                    # Mark first token
                    if first_token_time is None:
                        first_token_time = time.time()
                        timings['first_token'] = first_token_time - start_time
                        print(f"\n⏰ [{timings['first_token']:.2f}s] First token received")
                        print(f"🤖 LLM: ", end='', flush=True)

                    content = event.get("content", "")
                    print(content, end='', flush=True)
                    token_count += 1
                    last_token_time = time.time()

                elif event_type == "message":
                    message_time = time.time()
                    timings['message_received'] = message_time - start_time

                elif event_type == "done":
                    done_time = time.time()
                    timings['done'] = done_time - start_time
                    print(f"\n\n⏰ [{timings['done']:.2f}s] Stream completed")
                    break

                elif event_type == "error":
                    error_msg = event.get("error", "Unknown error")
                    print(f"\n\n❌ Error: {error_msg}")
                    return

            except json.JSONDecodeError:
                continue

        # Calculate statistics
        total_time = time.time() - start_time

        if first_token_time and last_token_time:
            generation_time = last_token_time - first_token_time
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0
        else:
            generation_time = 0
            tokens_per_second = 0

        # Print summary
        print("\n" + "="*80)
        print("⏱️  TIMING BREAKDOWN")
        print("="*80)
        print(f"  Request sent:        {timings.get('request_sent', 0):.3f}s")
        print(f"  First byte received: {timings.get('first_byte', 0):.3f}s  ← LATENCY")
        print(f"  First token:         {timings.get('first_token', 0):.3f}s  ← LLM START")
        print(f"  Stream completed:    {timings.get('done', total_time):.3f}s")
        print(f"  Total time:          {total_time:.3f}s")
        print("-"*80)
        print(f"  Tokens generated:    {token_count}")
        print(f"  Generation time:     {generation_time:.3f}s")
        print(f"  Speed:               {tokens_per_second:.1f} tokens/sec")
        print("="*80)

        # Identify bottleneck
        print("\n🔍 BOTTLENECK ANALYSIS:")

        if timings.get('first_byte', 0) > 5:
            print(f"  ⚠️  High network latency: {timings['first_byte']:.1f}s to first byte")
            print(f"     This is network/server delay, not your code!")

        if timings.get('first_token', 0) - timings.get('first_byte', 0) > 10:
            print(f"  ⚠️  LLM processing delay: {timings['first_token'] - timings.get('first_byte', 0):.1f}s before first token")
            print(f"     The LLM is thinking/processing your request")

        if tokens_per_second < 10 and token_count > 0:
            print(f"  ⚠️  Slow generation: {tokens_per_second:.1f} tokens/sec")
            print(f"     The LLM model itself is slow")

        if total_time < 5:
            print(f"  ✅ Response was fast ({total_time:.1f}s total)")
        elif total_time < 15:
            print(f"  ⚡ Response was acceptable ({total_time:.1f}s total)")
        else:
            print(f"  🐌 Response was slow ({total_time:.1f}s total)")

    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Status: {e.response.status_code}")
            print(f"   Body: {e.response.text[:500]}")
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out after 120 seconds")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  LLM PERFORMANCE DIAGNOSTIC TOOL")
    print("="*80)
    print("\nThis will show exactly where time is being spent.")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            prompt = input("\n💬 You: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break

            send_message(prompt)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")