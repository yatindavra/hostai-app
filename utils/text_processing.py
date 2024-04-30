import re


def number_to_words(num):
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    # Define the magnitude names
    magnitudes = [
        "",
        "thousand",
        "million",
        "billion",
        "trillion",
        "quadrillion",
        "quintillion",
        "sextillion",
        "septillion",
        "octillion",
        "nonillion",
        "decillion",
    ]

    def convert_below_1000(n):
        if n == 0:
            return ""
        elif n < 10:
            return ones[n]
        elif n < 20:
            return teens[n - 10]
        elif n < 100:
            return tens[n // 10] + " " + convert_below_1000(n % 10)
        else:
            return ones[n // 100] + " hundred " + convert_below_1000(n % 100)

    if num == 0:
        return "zero"

    num_chunks = []
    while num:
        num_chunks.append(num % 1000)
        num //= 1000

    words_chunks = [
        convert_below_1000(chunk) + " " + magnitudes[i]
        for i, chunk in enumerate(num_chunks)
    ]
    words = " ".join(words_chunks[::-1])

    return words.strip()


def replace_numbers_with_words(input_text):
    numeric_values = re.findall(r"\b\d+\b", input_text)
    for num in numeric_values:
        num_as_text = number_to_words(int(num))
        input_text = input_text.replace(num, num_as_text)
    return input_text
