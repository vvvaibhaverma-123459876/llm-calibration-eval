"""50-question trivia dataset with known answers."""

from __future__ import annotations

from llm_calibration_eval.providers.base import CalibrationSample

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TRIVIA_DATASET: list[CalibrationSample] = [
    CalibrationSample("What is the capital of France?", "Paris", "geography"),
    CalibrationSample("What is the chemical symbol for gold?", "Au", "science"),
    CalibrationSample("Who wrote 'Romeo and Juliet'?", "William Shakespeare", "literature"),
    CalibrationSample("What is the largest planet in our solar system?", "Jupiter", "science"),
    CalibrationSample("In what year did World War II end?", "1945", "history"),
    CalibrationSample("What is the square root of 144?", "12", "math"),
    CalibrationSample("Who painted the Mona Lisa?", "Leonardo da Vinci", "art"),
    CalibrationSample("What is the capital of Japan?", "Tokyo", "geography"),
    CalibrationSample("How many sides does a hexagon have?", "6", "math"),
    CalibrationSample("What is the fastest land animal?", "Cheetah", "science"),
    CalibrationSample("What is the currency of the United Kingdom?", "Pound sterling", "geography"),
    CalibrationSample("Who invented the telephone?", "Alexander Graham Bell", "history"),
    CalibrationSample("What is the boiling point of water in Celsius?", "100", "science"),
    CalibrationSample("In what continent is Egypt located?", "Africa", "geography"),
    CalibrationSample("What is the largest ocean on Earth?", "Pacific Ocean", "science"),
    CalibrationSample("Who was the first US President?", "George Washington", "history"),
    CalibrationSample("What is the chemical formula for water?", "H2O", "science"),
    CalibrationSample("How many bones are in the adult human body?", "206", "science"),
    CalibrationSample("What language is spoken in Brazil?", "Portuguese", "geography"),
    CalibrationSample("Who wrote '1984'?", "George Orwell", "literature"),
    CalibrationSample("What is the speed of light in km/s (approximate)?", "300000", "science"),
    CalibrationSample("What is the capital of Australia?", "Canberra", "geography"),
    CalibrationSample("Which element has atomic number 1?", "Hydrogen", "science"),
    CalibrationSample("What year did the Titanic sink?", "1912", "history"),
    CalibrationSample("Who composed the Fifth Symphony?", "Ludwig van Beethoven", "art"),
    CalibrationSample("What is 7 multiplied by 8?", "56", "math"),
    CalibrationSample("Which planet is known as the Red Planet?", "Mars", "science"),
    CalibrationSample("What is the longest river in the world?", "Nile", "geography"),
    CalibrationSample("In which country was Albert Einstein born?", "Germany", "history"),
    CalibrationSample("What is the capital of Canada?", "Ottawa", "geography"),
    CalibrationSample("How many planets are in our solar system?", "8", "science"),
    CalibrationSample("Who wrote 'Pride and Prejudice'?", "Jane Austen", "literature"),
    CalibrationSample("What is the smallest prime number?", "2", "math"),
    CalibrationSample("Which gas do plants absorb from the atmosphere?", "Carbon dioxide", "science"),
    CalibrationSample("What is the capital of Germany?", "Berlin", "geography"),
    CalibrationSample("In what year did man first land on the moon?", "1969", "history"),
    CalibrationSample("Who invented the light bulb?", "Thomas Edison", "history"),
    CalibrationSample("What is the chemical symbol for iron?", "Fe", "science"),
    CalibrationSample("What is the tallest mountain on Earth?", "Mount Everest", "geography"),
    CalibrationSample("What is the capital of Brazil?", "Brasilia", "geography"),
    CalibrationSample("What is 15% of 200?", "30", "math"),
    CalibrationSample("How many continents are there on Earth?", "7", "geography"),
    CalibrationSample("Who was the first person to walk on the moon?", "Neil Armstrong", "history"),
    CalibrationSample("What is the powerhouse of the cell?", "Mitochondria", "science"),
    CalibrationSample("Which Shakespeare play features the character Hamlet?", "Hamlet", "literature"),
    CalibrationSample("What is the atomic number of carbon?", "6", "science"),
    CalibrationSample("What is the capital of Russia?", "Moscow", "geography"),
    CalibrationSample("How many degrees are in a right angle?", "90", "math"),
    CalibrationSample("What is the most spoken language in the world?", "Mandarin Chinese", "geography"),
    CalibrationSample("What year was the Declaration of Independence signed?", "1776", "history"),
]


def load_dataset(name: str = "trivia") -> list[CalibrationSample]:
    """Load a built-in dataset by name.

    Args:
        name: Dataset name (currently only ``"trivia"`` is supported).

    Returns:
        List of :class:`CalibrationSample` objects.

    Raises:
        ValueError: If *name* is not a known dataset.
    """
    datasets: dict[str, list[CalibrationSample]] = {
        "trivia": TRIVIA_DATASET,
    }
    if name not in datasets:
        available = ", ".join(sorted(datasets))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return datasets[name]
