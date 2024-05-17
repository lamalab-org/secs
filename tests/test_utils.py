from molbind.utils.utils import find_all_pairs_in_list


def test_pairing_function():
    lst = [1, 2, 3, 4, 5]
    expected = [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
        (4, 5),
    ]
    assert find_all_pairs_in_list(lst) == expected
