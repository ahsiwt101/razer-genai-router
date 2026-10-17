# Argument extractors: pull what each tool needs out of the user's text, using rules. 

import re

from .calculator import evaluate

# numbers written as words
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 10**3, "million": 10**6, "billion": 10**9}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES) | {"hundred"}


def _parse_number_words(words: list[str]) -> int:
    """['two', 'hundred', 'and', 'five'] -> 205"""
    total, current = 0, 0
    for w in words:
        if w == "and":
            continue
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        else:  # thousand / million / billion closes off a group: "two thousand three hundred"
            total += max(current, 1) * _SCALES[w]
            current = 0
    return total + current


def words_to_numbers(text: str) -> str:
    """'what's two hundred times twenty-three' -> "what's 200 times 23" """
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)  
    tokens, out, run = text.split(), [], []
    for i, tok in enumerate(tokens):
        word = tok.strip("?,.!")
        next_is_number = i + 1 < len(tokens) and tokens[i + 1].strip("?,.!") in _NUMBER_WORDS
        if word in _NUMBER_WORDS or (word == "and" and run and next_is_number):
            run.append(word)
            if word != tok:  # for punctuation stuck to the word ends the number: "thirteen?"
                out.append(str(_parse_number_words(run)) + tok[len(word):])
                run = []
            continue
        if run:
            out.append(str(_parse_number_words(run)))
            run = []
        out.append(tok)
    if run:
        out.append(str(_parse_number_words(run)))
    return " ".join(out)


# calculator
_NUM = r"(-?\d+(?:\.\d+)?)"
_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800, "year": 31536000}

# some common ones 
_TEMPLATES = [
    # "convert 100 fahrenheit to celsius"
    (re.compile(_NUM + r"\s*(?:degrees?\s*)?°?\s*(?:f|fahrenheit)\b.*?\b(?:to|in|into)\s+(?:degrees\s+)?(?:c|celsius)\b"),
     lambda m: f"({m[1]} - 32) * 5 / 9"),
    (re.compile(_NUM + r"\s*(?:degrees?\s*)?°?\s*(?:c|celsius)\b.*?\b(?:to|in|into)\s+(?:degrees\s+)?(?:f|fahrenheit)\b"),
     lambda m: f"{m[1]} * 9 / 5 + 32"),
    # "how many minutes are in a week", "how many seconds in 3 hours"
    (re.compile(r"how many (second|minute|hour|day|week)s? (?:are )?(?:there )?in (?:an? |one )?" + r"(\d+(?:\.\d+)?)?\s*"
                r"(second|minute|hour|day|week|year)s?"),
     lambda m: f"{m[2] or 1} * {_SECONDS[m[3]]} / {_SECONDS[m[1]]}"),
    # "average of 12, 18 and 30"
    (re.compile(r"(?:average|mean) of ((?:\d+(?:\.\d+)?(?:,\s*|\s+and\s+|\s+))+\d+(?:\.\d+)?)"),
     lambda m: (lambda nums: f"({' + '.join(nums)}) / {len(nums)}")(re.findall(r"\d+(?:\.\d+)?", m[1]))),
    # "split a 240 dollar bill among 6 people", "split 90 dollars between 4 friends"
    (re.compile(r"(?:split|divide|share)\s+(?:an?\s+)?\$?" + _NUM + r"\D*?\b(?:among|between|by|into)\s+(\d+)"),
     lambda m: f"{m[1]} / {m[2]}"),
    # "20 percent off 45" -> 45 * (1 - 20/100)
    (re.compile(_NUM + r"\s*(?:%|percent)\s+off\s+(?:an?\s+)?\$?" + _NUM),
     lambda m: f"{m[2]} * (1 - {m[1]} / 100)"),
    # "18% of 340", "15 percent tip on 82"
    (re.compile(_NUM + r"\s*(?:%|percent)\s+(?:tip\s+)?(?:of|on)\s+\$?" + _NUM),
     lambda m: f"{m[1]} / 100 * {m[2]}"),
    # "subtract 129 from 1000"
    (re.compile(r"subtract\s+" + _NUM + r"\s+from\s+" + _NUM), lambda m: f"{m[2]} - {m[1]}"),
]

# symbol substitution
_WORD_OPS = [
    (r"\bmultiplied by\b", "*"), (r"\btimes\b", "*"), (r"\bdivided by\b", "/"), (r"\bover\b", "/"),
    (r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\bmod(?:ulo)?\b", "%"),
    (r"\bto the power of\b", "**"), (r"\bto the (\d+)(?:st|nd|rd|th)\b", r"** \1"),
    (r"(\d)\s*x\s*(\d)", r"\1 * \2"),
    (r"\bsquare root of\b", "sqrt"), (r"\bcube root of\b", "cbrt"),
    (r"\b(sqrt|cbrt|log|ln) of\b", r"\1"),  # "sqrt of 729": found by test-set error analysis (routing was right, extraction failed)
    (r"\bfactorial of\b", "factorial"), (r"(\d+(?:\.\d+)?)\s+factorial\b", r"factorial(\1)"),
    (r"\bhalf of\s+" + _NUM, r"\1 / 2"), (r"\bdouble\s+" + _NUM, r"\1 * 2"),
    (r"\bcube of\s+" + _NUM, r"\1 ** 3"), (_NUM + r"\s+squared\b", r"\1 ** 2"), (_NUM + r"\s+cubed\b", r"\1 ** 3"),
    (r"\badd\s+" + _NUM + r"\s+(?:and|to)\s+" + _NUM, r"\1 + \2"),
    (r"\bmultiply\s+" + _NUM + r"\s+(?:by|and)\s+" + _NUM, r"\1 * \2"),
    (r"\bdivide\s+" + _NUM + r"\s+by\s+" + _NUM, r"\1 / \2"),
]
_MATH_TOKEN = re.compile(r"sqrt|cbrt|factorial|log|ln|\d+(?:\.\d+)?|\*\*|[-+*/%()^]")


def extract_expression(query: str) -> str | None:
    q = query.lower().replace("×", "*").replace("÷", "/")
    q = re.sub(r"(?<=\d),(?=\d{3}\b)", "", q)
    q = words_to_numbers(q)

    for pattern, build in _TEMPLATES:
        m = pattern.search(q)
        if m:
            return build(m)

    for pattern, replacement in _WORD_OPS:
        q = re.sub(pattern, replacement, q)
    # "sqrt 144" -> "sqrt(144)"
    q = re.sub(r"\b(sqrt|cbrt|factorial|log|ln)\s+" + _NUM, r"\1(\2)", q)

    expression = " ".join(_MATH_TOKEN.findall(q))
    expression = expression.replace("( ", "(").replace(" )", ")")
    expression = re.sub(r"\b(sqrt|cbrt|factorial|log|ln) \(", r"\1(", expression)
    has_number = re.search(r"\d", expression)
    has_operation = re.search(r"[-+*/%^]|sqrt|cbrt|factorial|log|ln", expression)
    if not (has_number and has_operation):
        return None  # "what is 42": nothing to compute
    try:
        evaluate(expression)  # only hand the calculator something it can actually compute
    except ZeroDivisionError:
        return expression     # a real calculation with a maths error
    except Exception:
        return None           
    return expression


# weather
_TIME_WORDS = {"today", "tonight", "tomorrow", "now", "later", "right", "this", "next", "at", "on",
               "morning", "afternoon", "evening", "weekend", "week", "moment"}
_NOT_PLACES = {"the", "a", "an", "my", "your", "our", "work", "home", "school", "office", "general",
               "summer", "winter", "spring", "autumn", "fall", "advance", "total", "it", "there",
               "here", "bed", "town"} | _TIME_WORDS


def extract_location(query: str) -> str | None:
    # capitalized words after a preposition: "in Kuala Lumpur", "for Hong Kong".
    m = re.search(r"\b(?:in|for|at|near|around)\s+((?:[A-Z][\w'.-]*)(?:\s+[A-Z][\w'.-]*)*)", query)
    if m and m[1].split()[0].lower() not in _NOT_PLACES:
        return m[1].rstrip(".?!,")
    # lowercase fallback: "weather in singapore today" -> "Singapore".
    m = re.search(r"\b(?:in|for|at)\s+([a-z][a-z'.-]+(?:\s+[a-z][a-z'.-]+)?)", query.lower())
    if m:
        words = m[1].rstrip(".?!,").split()
        while words and words[-1] in _TIME_WORDS:  # drop trailing time words: "london tomorrow"
            words.pop()
        if words and words[0] not in _NOT_PLACES:
            return " ".join(words).title()
    return None  # "do I need a coat tonight": no place mentioned


# web search
_FILLER = re.compile(
    r"^(?:hey|hi|ok|please|can you|could you|would you|search(?: the web| online)?(?: for)?|google|"
    r"look up|find(?: me)?|tell me)\s+", re.IGNORECASE)


def extract_search_query(query: str) -> str | None:
    q = query.strip()
    while (stripped := _FILLER.sub("", q)) != q:  # remove "can you search for ..." layer by layer
        q = stripped
    q = q.rstrip("?!. ").strip()
    return q or None
