from pick import pick

class Menu:
    """A versatile class to create and display interactive CLI menus."""
    def __init__(self, title, options):
        """
        Initializes a Menu.
        Args:
            title (str): The title to display above the menu options.
            options (list): A list of tuples, where each tuple contains:
                            - str: The text to display for the option.
                            - function: The function to call when the option is selected.
        """
        self.title = title
        self.options = options
        self.option_map = {text: func for text, func in options}
        self.option_texts = [text for text, _ in options]

    def show(self):
        """
        Displays the menu and executes the callback for the selected option.
        The menu will loop until an option with the text 'Back' or 'Exit' is chosen.
        """
        while True:
            selected_text, _ = pick(self.option_texts, self.title, indicator='=>')

            # Exit condition for the menu loop
            if selected_text in ['Back', 'Exit']:
                return

            callback = self.option_map.get(selected_text)
            if callback:
                callback()  # Execute the associated function