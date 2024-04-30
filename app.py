from flask import request, jsonify, send_file, render_template, Flask
from utils.text_processing import replace_numbers_with_words
from utils.data_extraction import *
from utils.audio_generation import generate_audio
import json
import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from collections import deque
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import pandas as pd
import base64

app = Flask(__name__)

CORS(app, origins="*")

JSON_FOLDER = os.path.join(os.path.dirname(__file__), "JsonFiles")

WAV_FOLDER = os.path.join(os.path.dirname(__file__), "wavFiles")

USER_FOLDER = os.path.join(os.path.dirname(__file__), "User_Json")

ENQUIRY_FOLDER = os.path.join(os.path.dirname(__file__), "Enquiry_Json")

Call_Recording_FOLDER = os.path.join(os.path.dirname(__file__), "CallRecord_Json")

CURRENT_DIR = os.path.dirname(__file__)

empty_responses = deque(maxlen=3)

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

client = MongoClient(
    "mongodb+srv://nisujadhav657:bc5adVc5Btc5qfz4@hostinguser.kc0uvjw.mongodb.net/"
)
db = client["Health_Insurance"]

consecutive_negative_responses = 0

@app.route("/")
def render():
    return render_template("index.html")

@app.route("/health-insurance", methods=["POST"])
def health_insurance():
    data = request.json
    action = data.get("action")
    voice = data.get("voice")
    guid = request.headers.get("X-GUID")
    fileName = f"user_{guid}.json"
    user_json_path = os.path.join(USER_FOLDER, fileName)

    if voice == "male":
        voiceOFAi = os.path.join(CURRENT_DIR, "lib", "common", "orca_params_male.pv")
    else:
        voiceOFAi = os.path.join(CURRENT_DIR, "lib", "common", "orca_params_female.pv")

    if action == "send_audio":
        with open(os.path.join(JSON_FOLDER, "question.json"), "r") as file:
            questions = json.load(file)

        first_question = questions[0]

        question_to_copy = {
            "que_id": first_question["que_id"],
            "question": first_question["question"],
            "answer": "",
            "tag": first_question["tag"],
            "trial": first_question["trial"],
        }

        if not os.path.exists(user_json_path):
            with open(user_json_path, "w") as new_file:
                json.dump([question_to_copy], new_file, indent=4)

        with open(user_json_path, "r") as file:
            data = json.load(file)

        question = data[0]["question"]

        newText = replace_numbers_with_words(question)

        responsefilename = "response.wav"
        wav_path = os.path.join(WAV_FOLDER, responsefilename)

        generate_audio(newText, wav_path, voiceOFAi)

        with open(wav_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({"filename": responsefilename, "audio_data": audio_data})

    elif action == "receive_response":
        response = data.get("response")

        with open(user_json_path, "r") as file:
            data = json.load(file)

        u_questions = [q for q in data if q["answer"] == ""]

        if not u_questions:
            return jsonify({"message": "No unanswered questions left."})

        one_object = u_questions[0]
        que_id = one_object["que_id"]

        if not response.strip():
            empty_responses.append(response)

            if len(empty_responses) == 3 and all(
                not response.strip() for response in empty_responses
            ):
                text = "I think we have a weak connection. Our team will get back to you shortly. Thank you for your time. Have a great day."

                newText = replace_numbers_with_words(text)

                terminate_filename = "terminate.wav"

                wav_path = os.path.join(WAV_FOLDER, terminate_filename)

                generate_audio(newText, wav_path, voiceOFAi)

                with open(wav_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")

                return jsonify(
                    {"filename": terminate_filename, "audio_data": audio_data}
                )
            else:
                text = "Can't UnderStand. Are you there?"

                newText = replace_numbers_with_words(text)

                responsefilename = "response.wav"
                wav_path = os.path.join(WAV_FOLDER, responsefilename)

                generate_audio(newText, wav_path, voiceOFAi)

                with open(wav_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")

                return jsonify({"filename": responsefilename, "audio_data": audio_data})
        else:
            with open(os.path.join(JSON_FOLDER, "question.json"), "r") as file:
                dataOfQuestion = json.load(file)

            dataOfQUestionId = find_data_with_id(dataOfQuestion, que_id)
            validations = dataOfQUestionId["validation"]
            answer_validated = False
            for validation in validations:
                if validation == "text":
                    validate_name = extract_name(response)
                    if validate_name is not None:
                        one_object["answer"] = validate_name
                        answer_validated = True
                        break
                elif validation == "number":
                    vaidate_number = extract_numeric_value(response)
                    if vaidate_number is not None:
                        one_object["answer"] = round(vaidate_number)
                        answer_validated = True
                        break
                elif validation == "gender":
                    validate_gender = extract_gender(response)
                    if validate_gender is not None:
                        one_object["answer"] = validate_gender
                        answer_validated = True
                        break
                elif validation == "binary":
                    validate_binary = extract_binary_category(response)
                    if validate_binary is not None:
                        one_object["answer"] = validate_binary
                        answer_validated = True
                        break
                elif validation == "city":
                    cities_data = os.path.join(JSON_FOLDER, "cities.json")
                    validate_city = extract_cities_from_text(response, cities_data)
                    if validate_city is not None:
                        one_object["answer"] = validate_city
                        answer_validated = True
                        break
                elif validation == "job":
                    jobs_data = os.path.join(JSON_FOLDER, "jobs.json")
                    validate_job = extract_jobTitles(response, jobs_data)
                    if validate_job is not None:
                        one_object["answer"] = validate_job
                        answer_validated = True
                        break
                elif validation == "hearditary_dieces":
                    diseases_data = os.path.join(JSON_FOLDER, "dieasces.json")
                    validate_diecse = extract_hereditary_disease(
                        response, diseases_data
                    )
                    if validate_diecse is not None:
                        one_object["answer"] = validate_diecse
                        answer_validated = True
                        break

            if not answer_validated:
                one_object["answer"] = ""

            with open(user_json_path, "w") as file:
                json.dump(data, file, indent=4)

            with open(user_json_path, "r") as file:
                new_data = json.load(file)

            answeOfId = find_data_with_id(new_data, que_id)
            nextquestionID = find_data_with_id(dataOfQuestion, str(int(que_id) + 1))

            if nextquestionID is None:
                file_path = "insurance_data.csv"

                original_df = pd.read_csv(file_path)

                imputer = SimpleImputer(strategy="mean")

                column_with_missing_values = ["age"]

                original_df[column_with_missing_values] = imputer.fit_transform(
                    original_df[column_with_missing_values]
                )

                features = original_df.drop(["claim"], axis=1)
                target = original_df["claim"]

                categorical_columns = features.select_dtypes(include=["object"]).columns

                features = pd.get_dummies(
                    features, columns=categorical_columns, drop_first=True
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )

                reg = RandomForestRegressor(
                    n_estimators=400, max_depth=4, random_state=42
                )
                reg.fit(X_train, y_train)

                with open(user_json_path, "r") as file:
                    final_data = json.load(file)

                user_name = find_data_with_tag_and_get_answer(final_data, "name")
                user_age = find_data_with_tag_and_get_answer(final_data, "age")
                user_weight = find_data_with_tag_and_get_answer(final_data, "weight")
                user_city_name = find_data_with_tag_and_get_answer(final_data, "city")
                user_city = format_name(user_city_name)
                user_gender = find_data_with_tag_and_get_answer(final_data, "gender")
                user_job = find_data_with_tag_and_get_answer(final_data, "job")
                user_job_title = format_name(user_job)
                user_members = find_data_with_tag_and_get_answer(final_data, "members")
                user_dieces = find_data_with_tag_and_get_answer(final_data, "dieces")
                user_smoker = find_data_with_tag_and_get_answer(final_data, "smoke")
                user_bloodpressure = find_data_with_tag_and_get_answer(final_data, "bp")
                user_regular_ex = find_data_with_tag_and_get_answer(
                    final_data, "exercise"
                )

                if user_dieces == "diabetes":
                    user_diabetes = "yes"
                else:
                    user_diabetes = "no"

                check_dieases_lower = [disease.lower() for disease in user_dieces]

                matching_row = original_df[
                    (original_df["age"] == user_age)
                    & (original_df["sex"] == user_gender)
                    & (original_df["weight"] == user_weight)
                    & original_df["hereditary_diseases"].apply(
                        lambda x: any(
                            disease in x.lower() for disease in check_dieases_lower
                        )
                    )
                    & (original_df["members"] == user_members)
                    & (original_df["smoker"] == (1 if user_smoker == "yes" else 0))
                    & (original_df["city"] == user_city)
                    & (original_df["bloodpressure"] == user_bloodpressure)
                    & (original_df["diabetes"] == (1 if user_diabetes == "yes" else 0))
                    & (
                        original_df["regular_ex"]
                        == (1 if user_regular_ex == "yes" else 0)
                    )
                    & (original_df["job_title"] == user_job_title)
                ]

                if not matching_row.empty:
                    actual_charges = matching_row["claim"].min()

                    decimal_place = 2
                    rounded_charge = round(actual_charges, decimal_place)
                    text = f"So {user_name} Based on the information you provided, the estimated insurance charges for your tailored plan are: {rounded_charge}.Thank you {user_name} for co-operate with us and sharing all informations. It will help us find the best health insurance plan for you."

                else:
                    user_data = pd.DataFrame(
                        {
                            "age": [user_age],
                            "sex_male": [1 if user_gender == "male" else 0],
                            "weight": [user_weight],
                            "hereditary_diseases_NoDisease": [
                                1 if len(user_dieces) > 0 else 0
                            ],
                            "members": [user_members],
                            "smoker": [1 if user_smoker == "yes" else 0],
                            **{
                                f"{col}_{user_input}": 1
                                for col, user_input in zip(
                                    categorical_columns, [user_city, user_job_title]
                                )
                            },
                            "bloodpressure": [user_bloodpressure],
                            "diabetes": [1 if user_diabetes == "yes" else 0],
                            "regular_ex": [1 if user_regular_ex == "yes" else 0],
                        },
                        columns=features.columns,
                    )

                    predicted_charge = reg.predict(user_data)

                    rounded_charge = round(predicted_charge[0])

                    new_entry = pd.DataFrame(
                        {
                            "age": [user_age],
                            "sex": [user_gender],
                            "weight": [user_weight],
                            "hereditary_diseases": [
                                (
                                    "NoDisease"
                                    if user_dieces == "No Diesces"
                                    else ", ".join(user_dieces)
                                )
                            ][0].replace('"', ""),
                            "members": [user_members],
                            "smoker": [1 if user_smoker == "yes" else 0],
                            "city": [user_city],
                            "bloodpressure": [user_bloodpressure],
                            "diabetes": [1 if user_diabetes == "yes" else 0],
                            "regular_ex": [1 if user_regular_ex == "yes" else 0],
                            "job_title": [user_job_title],
                            "claim": [rounded_charge],
                        },
                        columns=original_df.columns,
                    )
                    updated_df = pd.concat([original_df, new_entry], ignore_index=True)

                    updated_df.to_csv(file_path, index=False)

                    predicted_charge_rf = reg.predict(features)
                    mae_rf = mean_absolute_error(target, predicted_charge_rf)
                    r2_rf = r2_score(target, predicted_charge_rf)

                text = f"So {user_name} Based on the information you provided, the estimated insurance charges for your tailored plan are: {rounded_charge}.Thank you {user_name} for co-operate with us and sharing all informations. It will help us find the best health insurance plan for you."

                newText = replace_numbers_with_words(text)

                tqfilename = "thankyou.wav"
                wav_path = os.path.join(WAV_FOLDER, tqfilename)

                generate_audio(newText, wav_path, voiceOFAi)

                with open(wav_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")

                return jsonify({"filename": tqfilename, "audio_data": audio_data})

            if answeOfId["answer"] == "":
                with open(user_json_path, "r+") as file:
                    trial_check = json.load(file)
                    dataForTrial = next(
                        (q for q in trial_check if q["que_id"] == que_id), None
                    )

                    if dataForTrial:
                        dataForTrial["trial"] += 1
                        file.seek(0)
                        json.dump(trial_check, file, indent=4)
                        file.truncate()

                        if dataForTrial["trial"] > 3:
                            text = "I think we have a weak connection. Our team will get back to you shortly. Thank you for your time. Have a great day."
                            newText = replace_numbers_with_words(text)

                            terminate_filename = "terminate.wav"

                            wav_path = os.path.join(WAV_FOLDER, terminate_filename)

                            generate_audio(newText, wav_path, voiceOFAi)

                            with open(wav_path, "rb") as f:
                                audio_data = base64.b64encode(f.read()).decode("utf-8")

                            return jsonify(
                                {
                                    "filename": terminate_filename,
                                    "audio_data": audio_data,
                                }
                            )
                        else:
                            text = dataOfQUestionId["vaidation_message"]
            else:
                text = nextquestionID["question"]
                question_to_copy = {
                    "que_id": nextquestionID["que_id"],
                    "question": nextquestionID["question"],
                    "answer": "",
                    "tag": nextquestionID["tag"],
                    "trial": nextquestionID["trial"],
                }

                new_data.append(question_to_copy)
                with open(user_json_path, "w") as file:
                    json.dump(new_data, file, indent=4)

            newText = replace_numbers_with_words(text)

            responsefilename = "response.wav"
            wav_path = os.path.join(WAV_FOLDER, responsefilename)

            generate_audio(newText, wav_path, voiceOFAi)

            with open(wav_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")

            return jsonify({"filename": responsefilename, "audio_data": audio_data})


@app.route("/enquiry", methods=["POST"])
def enquiry():
    data = request.json
    dataset_collection = db["enquiry_questions"]
    dataset = list(dataset_collection.find())
    voice = data.get("voice")
    action = data.get("action")
    conversation = []
    guid = request.headers.get("X-GUID")
    fileName = f"user_enquiry_{guid}.json"

    if voice == "male":
        voiceOFAi = os.path.join(CURRENT_DIR, "lib", "common", "orca_params_male.pv")
    else:
        voiceOFAi = os.path.join(CURRENT_DIR, "lib", "common", "orca_params_female.pv")

    if action == "send_enquiry_audio":
        text = "Welcome to our Health Insurance Planning System!.How can i assist you today?"

        newText = replace_numbers_with_words(text)

        responsefilename = "response.wav"
        wav_path = os.path.join(WAV_FOLDER, responsefilename)

        generate_audio(newText, wav_path, voiceOFAi)

        with open(wav_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({"filename": responsefilename, "audio_data": audio_data})

    elif action == "receive_enquiry_response":

        user_question = data.get("response")

        print("condition ", user_question.strip())
        if not user_question.strip():
            empty_responses.append(user_question)

            if len(empty_responses) == 3 and all(
                not response.strip() for response in empty_responses
            ):
                text = "I think we have a weak connection. Our team will get back to you shortly. Thank you for your time. Have a great day."

                newText = replace_numbers_with_words(text)

                terminate_filename = "terminate.wav"

                wav_path = os.path.join(WAV_FOLDER, terminate_filename)

                generate_audio(newText, wav_path, voiceOFAi)

                with open(wav_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")

                return jsonify(
                    {"filename": terminate_filename, "audio_data": audio_data}
                )
            else:
                text = "Can't UnderStand. Are you there?"

                newText = replace_numbers_with_words(text)

                responsefilename = "response.wav"
                wav_path = os.path.join(WAV_FOLDER, responsefilename)

                generate_audio(newText, wav_path, voiceOFAi)

                with open(wav_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")

                return jsonify({"filename": responsefilename, "audio_data": audio_data})
        else:
            empty_responses.clear()

            conversation.append({"user": user_question})
            user_question_embeding = model.encode(user_question, convert_to_tensor=True)
            similarities = []

            for data in dataset:
                dataset_question_embeding = model.encode(
                    data["question"], convert_to_tensor=True
                )
                similarity_score = util.pytorch_cos_sim(
                    user_question_embeding, dataset_question_embeding
                )
                similarities.append(similarity_score)

            most_similar_index = similarities.index(max(similarities))

            if similarities[most_similar_index] < 0.3:
                chatgpt_response = generate_chatgpt_response(user_question)

                newdata = {"question": user_question, "answer": chatgpt_response}
                dataset_collection.insert_one(newdata)

                text = chatgpt_response

            else:
                answer = dataset[most_similar_index]["answer"]
                if (
                    user_question.lower()
                    != dataset[most_similar_index]["question"].lower()
                ):
                    newdata = {"question": user_question, "answer": answer}
                    dataset_collection.insert_one(newdata)
                text = answer

            conversation.append({"AI": text})

            with open(os.path.join(ENQUIRY_FOLDER, fileName), "w") as f:
                json.dump(conversation, f)

            newText = replace_numbers_with_words(text)

            responsefilename = "response.wav"
            wav_path = os.path.join(WAV_FOLDER, responsefilename)

            generate_audio(newText, wav_path, voiceOFAi)

            with open(wav_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")

            return jsonify({"filename": responsefilename, "audio_data": audio_data})


@app.route("/callUser", methods=["POST"])
def callUser():
    data = request.json
    dataset_collection = db["calling_questions"]
    dataset = list(dataset_collection.find())
    voice = data.get("voice")
    action = data.get("action")
    conversation = []
    guid = request.headers.get("X-GUID")
    fileName = f"user_enquiry_{guid}.json"
    username = "thanos"
    tag = data.get("tag")
    global consecutive_negative_responses
    
    print('tag',tag)

    if voice == "male":
        voiceOFAi = os.path.join(CURRENT_DIR, "lib", "common", "orca_params_male.pv")
    else:
        voiceOFAi = os.path.join(CURRENT_DIR, "lib", "common", "orca_params_female.pv")

    if action == "send_call_audio":
        text = f"Hello {username}.my name is titan, and I'm calling from Health Insurance system. How are you today?"

        newText = replace_numbers_with_words(text)

        responsefilename = "response.wav"
        wav_path = os.path.join(WAV_FOLDER, responsefilename)

        generate_audio(newText, wav_path, voiceOFAi)

        with open(wav_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({"filename": responsefilename, "audio_data": audio_data})

    elif action == "recieve_call_response":
        user_response = data.get("response")

        negative_keywords = [
            "not interested",
            "don't want",
            "don't have time",
            "don't need",
            "don't",
        ]

        print("User Response", user_response)
        if any(keyword in user_response for keyword in negative_keywords):
            consecutive_negative_responses += 1
        else:
            consecutive_negative_responses = 0

        print("Negative Count", consecutive_negative_responses)
        print("Negative Count Condition ", consecutive_negative_responses >= 4)

        if not user_response.strip():
            empty_responses.append(user_response)

            if len(empty_responses) == 3 and all(
                not response.strip() for response in empty_responses
            ):
                text = "It seems we're having trouble with the connection.I will call you latter. Goodbye!"

                newText = replace_numbers_with_words(text)

                terminate_filename = "terminate.wav"

                wav_path = os.path.join(WAV_FOLDER, terminate_filename)

                generate_audio(newText, wav_path, voiceOFAi)

                with open(wav_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")

                return jsonify(
                    {"filename": terminate_filename, "audio_data": audio_data}
                )

            else:
                text = f"I couldn't hear you. Are you there {username}?"

                newText = replace_numbers_with_words(text)

                responsefilename = "response.wav"
                wav_path = os.path.join(WAV_FOLDER, responsefilename)

                generate_audio(newText, wav_path, voiceOFAi)

                with open(wav_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")

                return jsonify({"filename": responsefilename, "audio_data": audio_data})

        elif consecutive_negative_responses >= 4:

            text = f"Okay {username}. Thank you for your time. Have a great day!"

            newText = replace_numbers_with_words(text)

            terminate_filename = "terminate.wav"

            wav_path = os.path.join(WAV_FOLDER, terminate_filename)

            generate_audio(newText, wav_path, voiceOFAi)

            with open(wav_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")

            return jsonify({"filename": terminate_filename, "audio_data": audio_data})

        elif "schedule my call" in user_response and tag == None:
            text = "Of course, I'd be happy to schedule the call for you. Could you please let me know what time works best for you?"

            newText = replace_numbers_with_words(text)

            responsefilename = "response.wav"

            wav_path = os.path.join(WAV_FOLDER, responsefilename)

            generate_audio(newText, wav_path, voiceOFAi)

            with open(wav_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")

            return jsonify(
                {
                    "filename": responsefilename,
                    "audio_data": audio_data,
                    "tag": "schedule",
                }
            )
        elif tag == "schedule":
            schedule_time = extract_date_time(user_response)
            if schedule_time:
                text = f"Excellent! I've scheduled the call for {schedule_time} to discuss your health insurance plan. Thank you for your time. Have a great day {username}."
                responsefilename = "schedule.wav"

                wav_path = os.path.join(WAV_FOLDER, responsefilename)
                
                newText = replace_numbers_with_words(text)

                generate_audio(newText, wav_path, voiceOFAi)

                with open(wav_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")

                return jsonify({"filename": responsefilename, "audio_data": audio_data})
            else:
                text = "Can you please provide me valid date and time so i can easily schedule your call."

                newText = replace_numbers_with_words(text)

                responsefilename = "response.wav"

                wav_path = os.path.join(WAV_FOLDER, responsefilename)

                generate_audio(newText, wav_path, voiceOFAi)

                with open(wav_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")

                return jsonify({"filename": responsefilename, "audio_data": audio_data})

        else:
            empty_responses.clear()

            conversation.append({"user": user_response})

            user_question_embeding = model.encode(user_response, convert_to_tensor=True)
            similarities = []

            for data in dataset:
                dataset_question_embedding = model.encode(
                    data["user"], convert_to_tensor=True
                )
                similarity_score = util.pytorch_cos_sim(
                    user_question_embeding, dataset_question_embedding
                )
                similarities.append(similarity_score)
            most_similar_index = similarities.index(max(similarities))

            if similarities[most_similar_index] < 0.3:
                text = "I couldn't hear you properly. Can you please say that again?"

            else:
                answer = dataset[most_similar_index]["ai"]
                if user_response.lower() != dataset[most_similar_index]["user"].lower():
                    newdata = {"user": user_response, "ai": answer}
                    dataset_collection.insert_one(newdata)
                text = answer

            conversation.append({"AI": text})

            with open(os.path.join(Call_Recording_FOLDER, fileName), "w") as f:
                json.dump(conversation, f)

            newText = replace_numbers_with_words(text)

            responsefilename = "response.wav"

            wav_path = os.path.join(WAV_FOLDER, responsefilename)

            generate_audio(newText, wav_path, voiceOFAi)

            with open(wav_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")

            return jsonify({"filename": responsefilename, "audio_data": audio_data})


if __name__ == "__main__":
    app.run(debug=True)
