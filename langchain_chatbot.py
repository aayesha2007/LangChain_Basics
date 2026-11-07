import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
MODEL_NAME = "llama3.2:latest"

PROMPTS = {
    "general": (
        "You are a helpful college assistant. Answer clearly and briefly. "
        "If you do not know a college-specific fact, say so. "
        "Student question: {question}"
    ),
    "admissions": (
        "You are a college admissions assistant. Give practical next steps. "
        "Do not invent deadlines, fees, or eligibility rules. "
        "Admissions question: {question}"
    ),
    "courses": (
        "You are a course planning assistant. Explain learning concepts "
        "in accessible language. Course question: {question}"
    ),
    "library": (
        "You are a library assistant. Help with research, sources, and citations. "
        "Do not invent library hours or services. Library question: {question}"
    ),
    "student_support": (
        "You are a supportive student services assistant. Be respectful and "
        "suggest practical steps. Student question: {question}"
    ),
}


def build_chains(model_name=MODEL_NAME):
    llm = ChatOllama(model=model_name, temperature=0.2)
    parser = StrOutputParser()

    answer_chains = {
        name: ChatPromptTemplate.from_template(template) | llm | parser
        for name, template in PROMPTS.items()
    }

    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this text in 3 to 5 concise bullet points. "
        "Do not add facts.\n\nText:\n{text}"
    )

    qa_prompt = ChatPromptTemplate.from_template(
        "Answer using only the context below. If the answer is not in it, "
        "say the context does not provide the answer.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )

    return {
        "llm": llm,
        "answer_chains": answer_chains,
        "summarization_chain": summary_prompt | llm | parser,
        "question_answering_chain": qa_prompt | llm | parser,
    }


def append_csv(filename, columns, row):
    path = DATASET / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_result(query_id, query, scenario, response, model):
    timestamp = datetime.now(timezone.utc).isoformat()

    append_csv(
        "user_queries.csv",
        ["timestamp_utc", "query_id", "scenario", "chain_type", "query"],
        {
            "timestamp_utc": timestamp,
            "query_id": query_id,
            "scenario": scenario,
            "chain_type": "main",
            "query": query,
        },
    )

    append_csv(
        "chatbot_responses.csv",
        ["timestamp_utc", "query_id", "model", "response"],
        {
            "timestamp_utc": timestamp,
            "query_id": query_id,
            "model": model,
            "response": response,
        },
    )

    append_csv(
        "chain_results.csv",
        ["timestamp_utc", "query_id", "chain_type", "prompt_name", "result"],
        {
            "timestamp_utc": timestamp,
            "query_id": query_id,
            "chain_type": "main",
            "prompt_name": scenario,
            "result": response,
        },
    )

    append_csv(
        "prompts.csv",
        ["timestamp_utc", "prompt_name", "template"],
        {
            "timestamp_utc": timestamp,
            "prompt_name": scenario,
            "template": PROMPTS[scenario],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--scenario",
        choices=PROMPTS.keys(),
        default="general",
    )
    parser.add_argument("--question")
    args = parser.parse_args()

    chains = build_chains(args.model)

    print("College Assistant is ready. Type 'quit' to exit.")

    while True:
        question = args.question or input("\nYou: ").strip()

        if question.lower() in {"quit", "exit"}:
            break

        if question:
            query_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            try:
                response = chains["answer_chains"][args.scenario].invoke(
                    {"question": question}
                )
                print("\nAssistant:", response)
                log_result(
                    query_id,
                    question,
                    args.scenario,
                    response,
                    args.model,
                )
            except Exception as error:
                print("Could not get a response:", error)

        if args.question:
            break


if __name__ == "__main__":
    main()