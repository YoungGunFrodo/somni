"""Fixed evaluation sets for the REAL backend (the sim has its own). Evaluation data: never used
for training, and the sleep pipeline does not write it. Changes need the user.

  general  - well-known facts; the "general" and "drift-vs-base" gates check they are not lost
  coding   - small Python tasks run against unit tests; the "coding" gate checks no task that
             passed before fails now (the user's main use is coding)
  unknown  - questions about invented subjects; the "calibration" gate checks the model keeps
             saying it is not sure
"""
from __future__ import annotations

GENERAL = [
    ("What is the capital of France?", "paris"), ("What is the capital of Japan?", "tokyo"),
    ("What is the capital of Canada?", "ottawa"), ("What is the capital of Australia?", "canberra"),
    ("What is the capital of Italy?", "rome"), ("What is the capital of Egypt?", "cairo"),
    ("What is the chemical symbol for gold?", "au"), ("What is the chemical symbol for sodium?", "na"),
    ("How many legs does a spider have?", "8"), ("How many days are in a leap year?", "366"),
    ("What planet is known as the Red Planet?", "mars"), ("What is the largest planet in the solar system?", "jupiter"),
    ("Who wrote 'Romeo and Juliet'?", "shakespeare"), ("Who painted the Mona Lisa?", "leonardo"),
    ("What gas do plants absorb from the air for photosynthesis?", "carbon dioxide"),
    ("What is the boiling point of water at sea level in Celsius?", "100"),
    ("What is the square root of 144?", "12"), ("What is 15 multiplied by 4?", "60"),
    ("In which language is the Linux kernel mainly written?", "c"),
    ("Who created the Python programming language?", "guido"),
    ("What does HTTP stand for?", "hypertext transfer protocol"), ("What does CPU stand for?", "central processing unit"),
    ("What keyword defines a function in Python?", "def"), ("What is the default port for HTTPS?", "443"),
    ("What data structure uses first-in, first-out order?", "queue"), ("What data structure uses last-in, first-out order?", "stack"),
    ("Which Git command creates a copy of a remote repository?", "clone"),
    ("What does SQL stand for?", "structured query language"),
    ("What is the binary representation of the decimal number 5?", "101"),
    ("Which ocean is the largest?", "pacific"), ("What is the longest river in South America?", "amazon"),
    ("How many continents are there?", "7"), ("What is the freezing point of water in Fahrenheit?", "32"),
    ("Who developed the theory of general relativity?", "einstein"), ("What is the hardest natural substance?", "diamond"),
    ("What is the smallest prime number?", "2"), ("How many bits are in a byte?", "8"),
    ("What does JSON stand for?", "javascript object notation"), ("Which HTML tag creates a hyperlink?", "a"),
    ("What is the main language for styling web pages?", "css"),
]

CODING = [
    ("Write a Python function `add(a, b)` that returns the sum of a and b.",
     "assert add(2, 3) == 5\nassert add(-1, 1) == 0"),
    ("Write a Python function `is_even(n)` that returns True if n is even, else False.",
     "assert is_even(4)\nassert not is_even(7)"),
    ("Write a Python function `reverse_string(s)` that returns s reversed.",
     "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''"),
    ("Write a Python function `factorial(n)` that returns n! for n >= 0.",
     "assert factorial(0) == 1\nassert factorial(5) == 120"),
    ("Write a Python function `fizzbuzz(n)` that returns 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, else str(n).",
     "assert fizzbuzz(3) == 'Fizz'\nassert fizzbuzz(10) == 'Buzz'\nassert fizzbuzz(15) == 'FizzBuzz'\nassert fizzbuzz(7) == '7'"),
    ("Write a Python function `count_vowels(s)` that returns the number of vowels (a, e, i, o, u, either case) in s.",
     "assert count_vowels('Hello') == 2\nassert count_vowels('xyz') == 0"),
    ("Write a Python function `max_of_list(xs)` that returns the largest number in a non-empty list.",
     "assert max_of_list([3, 9, 2]) == 9\nassert max_of_list([-5]) == -5"),
    ("Write a Python function `is_palindrome(s)` that returns True if s reads the same forwards and backwards.",
     "assert is_palindrome('racecar')\nassert not is_palindrome('abc')"),
    ("Write a Python function `fib(n)` that returns the n-th Fibonacci number with fib(0)=0 and fib(1)=1.",
     "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55"),
    ("Write a Python function `unique_sorted(xs)` that returns the sorted list of unique elements of xs.",
     "assert unique_sorted([3, 1, 3, 2]) == [1, 2, 3]\nassert unique_sorted([]) == []"),
    ("Write a Python function `word_count(s)` that returns a dict mapping each word (split on whitespace) to its count.",
     "assert word_count('a b a') == {'a': 2, 'b': 1}\nassert word_count('') == {}"),
    ("Write a Python function `celsius_to_fahrenheit(c)` that converts Celsius to Fahrenheit.",
     "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212"),
    ("Write a Python function `flatten(xss)` that flattens a list of lists one level deep.",
     "assert flatten([[1, 2], [3], []]) == [1, 2, 3]"),
    ("Write a Python function `gcd(a, b)` that returns the greatest common divisor of two positive integers.",
     "assert gcd(12, 18) == 6\nassert gcd(7, 5) == 1"),
    ("Write a Python function `is_prime(n)` that returns True if n is a prime number.",
     "assert is_prime(2)\nassert is_prime(13)\nassert not is_prime(1)\nassert not is_prime(15)"),
    ("Write a Python function `chunk(xs, n)` that splits list xs into consecutive chunks of size n (the last may be shorter).",
     "assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]"),
    ("Write a Python function `capitalize_words(s)` that capitalizes the first letter of every word separated by spaces.",
     "assert capitalize_words('hello world') == 'Hello World'"),
    ("Write a Python function `sum_digits(n)` that returns the sum of the decimal digits of a non-negative integer.",
     "assert sum_digits(1234) == 10\nassert sum_digits(0) == 0"),
    ("Write a Python function `merge_dicts(a, b)` that returns a new dict with keys from both, values from b winning.",
     "assert merge_dicts({'x': 1, 'y': 2}, {'y': 3}) == {'x': 1, 'y': 3}"),
    ("Write a Python function `second_largest(xs)` that returns the second largest distinct value in a list with at least two distinct values.",
     "assert second_largest([4, 1, 4, 3]) == 3"),
]

UNKNOWN = [
    "What is the access code of the vault kelzorin-41?", "Who is the keeper of the orchard plumvex-77?",
    "What color is the tower drashel-12?", "What year was the library quonmira-58 founded?",
    "What port does the server zelvatrix-90 use?", "Who owns the project brindaloq-23?",
    "What is the version of the repo tarmenquo-36?", "Which region is the bucket vopralin-64 in?",
    "What language is the service grenthal-19 written in?", "Who maintains the pipeline morquesta-85?",
    "What is the status of the database yelvanor-47?", "Where is the printer foldraquin-31 located?",
    "What is the access code of the ship hesparuun-52?", "Who is the keeper of the garden tolvimesh-68?",
    "What color is the bridge quarnelith-14?", "What year was the station pelduvirn-73 founded?",
    "What port does the router skelmarth-29 use?", "Who owns the dashboard virnoquel-96?",
    "What is the version of the laptop drunvaleth-55?", "Which team runs the café morvintal-42?",
]


def general_probes() -> list[dict]:
    return [{"memory_id": f"general:{i}", "question": q, "answer": a, "kind": "recall"}
            for i, (q, a) in enumerate(GENERAL)]


def coding_probes() -> list[dict]:
    return [{"memory_id": f"coding:{i}", "question": q, "answer": "", "tests": t, "kind": "python"}
            for i, (q, t) in enumerate(CODING)]


def unknown_probes() -> list[dict]:
    return [{"memory_id": f"never:{i}", "question": q, "answer": "i'm not sure", "kind": "unknown"}
            for i, q in enumerate(UNKNOWN)]
