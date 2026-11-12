"""Historical labeler. This file does not call a model API.

An earlier draft of this appendix labeled retrieved products with a paid
language-model API and stored the labels outside the repository. That client
is not included here, and this module refuses to run. The book does not
repeat that labeling step.

The sampling rule used to choose queries is implemented in
llm_labels_convert.py so a reader who already has a label file can check
whether the file's stored parameters reconstruct the sample. The published
command-line default of that labeler was 2,000 queries. The protocol's
sampling design names 10,000. Use the parameters stored in the file.
"""

from __future__ import annotations

import sys


def main() -> None:
    raise SystemExit(
        "This historical labeler is not runnable. It called a paid API. "
        "Use llm_labels_convert.py to read a label file you already have."
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
