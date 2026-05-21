import argparse
import json
from transformers import AutoTokenizer


def build_messages(record):
    if "messages" in record:
        return record["messages"]

    if "prompt" in record and "response" in record:
        return [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]

    if "instruction" in record and "output" in record:
        user = record["instruction"]
        if record.get("input"):
            user = user + "\n\n" + record["input"]
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": record["output"]},
        ]

    if "question" in record and "answer" in record:
        return [
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": record["answer"]},
        ]

    raise ValueError(f"Unsupported record keys: {list(record.keys())}")


def fallback_chat_text(messages):
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    n = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue

            record = json.loads(line)
            messages = build_messages(record)

            if getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                text = fallback_chat_text(messages)

            out = {
                "id": record.get("id", f"sample_{idx:06d}"),
                "source": record.get("source", "alignment"),
                "behavior_type": record.get("behavior_type", "unknown"),
                "prompt_harm": record.get("prompt_harm", "unknown"),
                "text": text,
                "messages": messages,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} examples to {args.output}")


if __name__ == "__main__":
    main()
