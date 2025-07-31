from gemini_query import answer_query

if __name__ == "__main__":
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print(answer_query(q))