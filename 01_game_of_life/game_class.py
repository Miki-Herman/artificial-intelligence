
class GameOfLife:
    def __init__(self):
        self.map = {
            0 : "-",
            1 : "#"
        }

    def print_row(self, row: list):
        """
        Prints a string representation of a generation of cells
        :param row: A list of representing a generation of cells
        :return: None
        """
        mapped_row = [self.map[cell] for cell in row]
        printable_row = " | ".join(mapped_row)

        print(printable_row)
        return

    @staticmethod
    def step(row: list) -> list:
        """
        0 -> means dead cell, 1 -> alive cell
        Rules:
            - Neighbouring boxes of a cell are like this --> X X O X X (0 -> a living cell, X -> neighbouring places)
            - If cell has exactly two neighbouring cells then it survives
            - If an empty cell has exactly two neighbouring cells then it's born
            - Else cell dies

        :param row: A list of representing a generation of cells
        :return: A list of new generation
        """

        new_generation = []
        row_len = len(row)

        for index in  range(row_len):
            # get neighbouring cells
            start = max(0, index - 2)
            end = min(row_len, index + 3)
            neighbouring_cells = row[start: end]

            # check if cell will be born
            if row[index] == 0 and sum(neighbouring_cells) == 2:
                new_generation.append(1)

            # check if cell will survive
            elif row[index] == 1 and sum(neighbouring_cells) == 3: # sum must be 3 because the checked cell has value of 1
                new_generation.append(1)

            # else cell dies
            else:
                new_generation.append(0)

        return new_generation

    def start_generation(self, row: list, times: int) -> None:
        """
        :param row: A list of representing the first generation of cells
        :param times: How many generations should be generated
        :return: None
        """
        # print first generation
        self.print_row(row)

        for i in range(times):
            row = self.step(row)
            self.print_row(row)

        return None
