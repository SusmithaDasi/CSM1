from app.graph import build_graph


def main():
    graph = build_graph()

    question = input("\nAsk a research question: ")

    initial_state = {
        "question": question,
        "subtasks": [],
        "worker_outputs": {},
        "critique": "",
        "final_answer": "",
        "needs_revision": False,
        "revision_count": 0,
    }

    print("\nRunning research pipeline...\n")

    result = graph.invoke(initial_state)

    print("=" * 60)
    print("SUBTASKS")
    print("=" * 60)
    for i, task in enumerate(result["subtasks"], start=1):
        print(f"{i}. {task}")

    print("\n" + "=" * 60)
    print("WORKER OUTPUTS")
    print("=" * 60)
    for task, output in result["worker_outputs"].items():
        print(f"\nTask: {task}")
        print("-" * 40)
        print(output["answer"])

        print("\nSources:")
        print(output["sources"])

    print("\n" + "=" * 60)
    print("CRITIC REVIEW")
    print("=" * 60)
    print(result["critique"])

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["final_answer"])


if __name__ == "__main__":
    main()