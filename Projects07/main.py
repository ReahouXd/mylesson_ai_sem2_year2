import re

class PythonEduBot:
    def __init__(self):
        self.commands = {
            "introduction": self.introduction,
            "install python": self.install_python,
            "hello world": self.hello_world,
            "data types": self.data_types,
            "functions": self.functions,
            "exercise": self.exercise,
            "help": self.help
        }

    def introduction(self):
        return ("Python is a high-level, interpreted programming language known for its simplicity and readability. "
                "It's widely used for web development, data analysis, artificial intelligence, scientific computing, and more.")

    def install_python(self):
        return ("You can install Python by downloading it from the official website [python.org](https://www.python.org/). "
                "Choose the version that matches your operating system and follow the installation instructions.")

    def hello_world(self):
        return ("Writing a 'Hello, World!' program in Python is simple. Just open your text editor or Python IDE and type the following code:\n"
                "```python\n"
                "print('Hello, World!')\n"
                "```\n"
                "Save the file with a `.py` extension and run it. You'll see the message 'Hello, World!' printed on the screen.")

    def data_types(self):
        return ("Python has several basic data types including:\n"
                "- **int**: for integers (e.g., 1, 2, 3)\n"
                "- **float**: for floating-point numbers (e.g., 1.5, 2.75)\n"
                "- **str**: for strings (e.g., 'hello', 'Python')\n"
                "- **list**: for ordered collections of items (e.g., [1, 2, 3])\n"
                "- **dict**: for key-value pairs (e.g., {'name': 'Alice', 'age': 25})\n"
                "- **bool**: for Boolean values (True or False)")

    def functions(self):
        return ("A function in Python is a reusable block of code that performs a specific task. "
                "You define a function using the `def` keyword, followed by the function name and parentheses. Here's an example:\n"
                "```python\n"
                "def greet(name):\n"
                "    return f'Hello, {name}!!'\n"
                "```\n"
                "You can call this function by passing an argument, like `greet('Alice')`, and it will return 'Hello, Alice!'.")

    def exercise(self):
        return ("### Practical Assignment\n"
                "Write a Python program that asks the user for their name and age, then prints a personalized greeting and "
                "calculates the year they were born. Here is a template to get you started:\n"
                "```python\n"
                "def main():\n"
                "    name = input('Enter your name: ')\n"
                "    age = int(input('Enter your age: '))\n"
                "    birth_year = 2024 - age\n"
                "    print(f'Hello, {name}! You were born in {birth_year}.')\n"
                "\n"
                "main()\n"
                "```\n"
                "Run this program and enter your details to see the result.")

    def help(self):
        return ("### Help\n"
                "You can ask me about the following topics:\n"
                "- `introduction`: Learn what Python is.\n"
                "- `install python`: How to install Python.\n"
                "- `hello world`: Writing a 'Hello, World!' program.\n"
                "- `data types`: Basic data types in Python.\n"
                "- `functions`: Understanding and creating functions.\n"
                "- `exercise`: Practical exercise to apply what you've learned.\n"
                "Just type the topic name to get started.")

    def handle_input(self, user_input):
        user_input = user_input.lower().strip()
        for command in self.commands:
            if re.search(r'\b' + command + r'\b', user_input):
                return self.commands[command]()
        return "I'm sorry, I don't understand that command. Type 'help' to see what I can do."

def main():
    bot = PythonEduBot()
    print("Hello! I'm your Python Educational Assistance Bot. Type 'help' to see what I can do.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Happy coding!")
            break
        response = bot.handle_input(user_input)
        print(f"\nBot: {response}")

if __name__ == "__main__":
    main()
