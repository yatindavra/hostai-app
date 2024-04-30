import json
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datetime import datetime, timedelta
from dateutil import parser


def extract_numeric_value(user_input):
    numeric_values = []
    numeric_str = ""
    number_mapping = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    for word in user_input.lower().split():
        if word.isdigit() or word.replace(".", "", 1).isdigit():
            numeric_str += word
        elif word in number_mapping:
            numeric_str += number_mapping[word]
        elif numeric_str:
            numeric_values.append(numeric_str)
            numeric_str = ""

    if numeric_str:
        numeric_values.append(numeric_str)
    if not numeric_values:
        return None
    try:
        last_sequence = numeric_values[-1]
        numeric_value = (
            float(last_sequence) if "." in last_sequence else int(last_sequence)
        )
        return numeric_value
    except ValueError:
        return None


def extract_name(input_text):
    matches_pattern = re.findall(r"my name is (\w+)", input_text, re.IGNORECASE)
    numeric_pattern = bool(re.search(r"\d", input_text))

    if numeric_pattern:
        return None
    elif matches_pattern:
        return matches_pattern[0]
    else:
        matches_words = re.findall(r"\b\w+\b", input_text)
        valid_names = [
            match
            for match in matches_words
            if match.lower() not in ["is", "my", "name"] and len(match) > 1
        ]

        if valid_names:
            return valid_names[0]
        else:
            return None


def format_name(name):
    format_word = "".join(word.capitalize() for word in name.split())
    return format_word


def find_data_with_id(data_list, target_id):
    for item in data_list:
        if item.get("que_id") == target_id:
            return item
    return None


def find_data_with_tag_and_get_answer(data_list, target_tag):
    for item in data_list:
        if item.get("tag") == target_tag:
            answer = item["answer"]
            return answer
    return None


def extract_gender(user_input):

    if "male" or "mail" in user_input:
        gender = "male"
    elif "female" in user_input:
        gender = "female"
    else:
        return None

    return gender


def extract_binary_category(user_input):

    if "yes" in user_input or "no" in user_input:
        last_occurrence_yes = user_input.rfind("yes")
        last_occurrence_no = user_input.rfind("no")
        last_occurrence = max(last_occurrence_yes, last_occurrence_no)
        category_value = user_input[last_occurrence : last_occurrence + 3]
    else:
        return None

    return category_value


def extract_cities_from_text(text, json_file, encoding="utf-8"):
    with open(json_file, "r", encoding=encoding) as file:
        data = json.load(file)
        city_names = data["cities"]

    cities_found = []
    for city in city_names:
        if re.search(r"\b{}\b".format(city), text, re.IGNORECASE):
            cities_found.append(city)

    if len(cities_found) > 0:
        return cities_found[-1]
    else:
        return None


def extract_jobTitles(text, json_file, encoding="utf-8"):
    with open(json_file, "r", encoding=encoding) as file:
        data = json.load(file)
        jobs = data["job_titles"]

    jobs_found = []
    for job in jobs:
        if re.search(r"\b{}\b".format(re.escape(job)), text, re.IGNORECASE):
            jobs_found.append(job)

    if len(jobs_found) > 0:
        return jobs_found[-1]
    else:
        return None


def extract_hereditary_disease(user_input, json_file, encoding="utf-8"):
    with open(json_file, "r", encoding=encoding) as file:
        data = json.load(file)
        hereditary_diseases = data["hereditary_diseases"]

    user_diseases = []
    if (
        "no" in user_input.lower()
        or "don't" in user_input.lower()
        or "donot" in user_input.lower()
    ):
        return "No Diseases"
    

    for disease in hereditary_diseases:
        if re.search(r"\b{}\b".format(re.escape(disease)), user_input, re.IGNORECASE):
            user_diseases.append(disease)
            return user_diseases

    return None

def generate_chatgpt_response(user_input):
    chatgpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    chatgpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_ids = chatgpt_tokenizer.encode(user_input, return_tensors="pt")
    output = chatgpt_model.generate(
        input_ids,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    response = chatgpt_tokenizer.decode(output[0], skip_special_tokens=True)

    return response

def extract_date_time(sentence):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    sentence = sentence.replace(
        "tomorrow", (today + timedelta(days=1)).strftime("%Y-%m-%d")
    )

    weekdays = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    for day in weekdays:
        if day in sentence.lower():
            day_index = weekdays.index(day)
            days_until_day = (day_index - today.weekday() + 7) % 7
            target_date = today + timedelta(days=days_until_day)
            sentence = sentence.replace(day, target_date.strftime("%Y-%m-%d"))
            break

    parsed_date_time = parser.parse(sentence, fuzzy=True)

    if parsed_date_time < datetime.now():
        return None

    return parsed_date_time