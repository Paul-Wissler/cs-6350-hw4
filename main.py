import numpy as np
import pandas as pd

from pathlib import Path

import QuestionAnswers.part2 as part2

pd.options.mode.chained_assignment = None  # default='warn'


def main():
    part2.q2a()
    part2.q2b()
    part2.q3a()
    part2.q3b_c()


if __name__ == '__main__':
    main()
