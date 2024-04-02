from HF_Mistral import hf_mistral

def main():
    question=input("Ask: ")
    n=int(input("length: "))
    result=hf_mistral(question=question, n=n)
    print(result)

if __name__ == "__main__":
    main()
