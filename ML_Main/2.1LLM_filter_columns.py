import csv
import requests

def query_llm(content):
    while True:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "qwen2.5-27b-instruct",
                "messages": [
                    {"role": "system", "content": "Always answer in Yes or No, Give right answer as possible."},
                    {"role": "user", "content": f'Answer in Yes or No: "{content}" 是否是下列参数其中之一的文字上相关的测量结果：PH，浊度，电导率，大肠杆菌，溶解氧，温度？Answer in Yes or No'}
                ],
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": False
            }
        )
        answer = response.json()["choices"][0]["message"]["content"].strip().lower()
        print(answer+'  '+content)
        if answer in ["yes", "no"]:
            return answer == "yes"


def filter_csv(input_filepath, output_filepath):
    with open(input_filepath, mode='r', encoding='utf-8') as infile, open(output_filepath, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        header = next(reader)
        writer.writerow(header)
        
        for row in reader:
            content = ','.join(row)
            if query_llm(content):
                print(row)
                writer.writerow(row)

if __name__ == "__main__":
    input_filepath = "./extracted_columns.csv"
    output_filepath = "./filtered_columns.csv"
    filter_csv(input_filepath, output_filepath)
