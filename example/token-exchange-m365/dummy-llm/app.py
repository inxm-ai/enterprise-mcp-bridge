from flask import Flask, request, jsonify, Response
import json
import uuid
import time

app = Flask(__name__)

@app.route('/', defaults={'path': ''}, methods=['POST', 'OPTIONS'])
@app.route('/<path:path>', methods=['POST', 'OPTIONS'])
def mirror_request(path):
    json_data = request.get_json(silent=True)
    if json_data:
        def generate_sse():
            if json_data["messages"] and json_data["messages"][-1].get("role") == "tool":
                # If the JSON includes a tool call, reply as assistant with the tool call reply
                tool_call_reply = json_data["messages"][-1].get("content")
                parsed_reply = json.loads(tool_call_reply)
                folder_list = json.loads(parsed_reply[0]["text"])["value"]
                markdown_list = "\n".join([f"- {folder['displayName']}" for folder in folder_list])

                event_data = {
                    "id": str(uuid.uuid4()),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
                    "choices": [
                        {"index": 0, "delta": {"content": "I'm not an llm, only a dummy implementation. I can only show you your inbox folders:\n" + markdown_list}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 33,
                        "total_tokens": 254,
                        "completion_tokens": 221
                    }
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                yield "data: [DONE]\n\n"
            elif len(json_data["messages"]) == 1:
                # If the JSON message has one entry, reply with an OpenAI-compatible tool call
                tool_call = [
                    {"index": 0, "function": {"name": "list-mail-folders", "arguments": "{}"}}
                ]
                event_data = {
                    "id": str(uuid.uuid4()),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
                    "choices": [
                        {"index": 0, "delta": { "tool_calls": tool_call}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 33,
                        "total_tokens": 254,
                        "completion_tokens": 221
                    }
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                # Return the tool call as content in the response when there's no tool call
                event_data = {
                    "id": str(uuid.uuid4()),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
                    "choices": [
                        {"index": 0, "delta": {"content": json_data}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 33,
                        "total_tokens": 254,
                        "completion_tokens": 221
                    }
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                yield "data: [DONE]\n\n"

        return Response(generate_sse(), content_type='text/event-stream')

    return jsonify({
        "method": request.method,
        "path": request.path,
        "headers": dict(request.headers),
        "args": request.args,
        "form": request.form,
        "json": json_data,
        "data": request.data.decode('utf-8')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8765)