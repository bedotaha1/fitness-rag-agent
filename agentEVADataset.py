
agent_golden_dataset = [

    # -------------------------
    # Should call the tool
    # -------------------------

    {
        "question": "How many grams of protein per kilogram of body weight per day are recommended for people engaged in resistance training?",
        "should_call_tool": True,
        "expected_answer_contains": "1.6 to 2.2"
    },
    {
        "question": "What's the recommended rest time between hypertrophy sets?",
        "should_call_tool": True,
        "expected_answer_contains": None
    },
    {
        "question": "How much sleep should athletes get for recovery?",
        "should_call_tool": True,
        "expected_answer_contains": None
    },
    {
        "question": "Is creatine safe to take every day?",
        "should_call_tool": True,
        "expected_answer_contains": None
    },
    {
        "question": "What's the difference between steady-state cardio and HIIT?",
        "should_call_tool": True,
        "expected_answer_contains": None
    },
    {
        "question": "How can I prevent shoulder injuries while bench pressing?",
        "should_call_tool": True,
        "expected_answer_contains": None
    },

    # -------------------------
    # Ambiguous -> clarify first
    # -------------------------

    {
        "question": "How many grams should I consume?",
        "should_call_tool": False,
        "expected_answer_contains": None
    },
    {
        "question": "How long should I rest?",
        "should_call_tool": False,
        "expected_answer_contains": None
    },
    {
        "question": "Is it good for me?",
        "should_call_tool": False,
        "expected_answer_contains": None
    },
    {
        "question": "Should I do more of it?",
        "should_call_tool": False,
        "expected_answer_contains": None
    },

    # -------------------------
    # Conversation
    # -------------------------

    {
        "question": "Hi!",
        "should_call_tool": False,
        "expected_answer_contains": None
    },
    {
        "question": "Thanks!",
        "should_call_tool": False,
        "expected_answer_contains": None
    },
    {
        "question": "Tell me a joke.",
        "should_call_tool": False,
        "expected_answer_contains": None
    },

    # -------------------------
    # Out of scope
    # -------------------------

    {
        "question": "What brewing temperature is recommended for pour-over coffee?",
        "should_call_tool": False,
        "expected_answer_contains": None
    },
    {
        "question": "Who invented the bicycle?",
        "should_call_tool": False,
        "expected_answer_contains": None
    },
    {
        "question": "Why do anglerfish have a glowing lure?",
        "should_call_tool": False,
        "expected_answer_contains": None
    },
    {
        "question": "How does sourdough fermentation work?",
        "should_call_tool": False,
        "expected_answer_contains": None
    },
    {
        "question": "What's the capital of Brazil?",
        "should_call_tool": False,
        "expected_answer_contains": None
    },
]
