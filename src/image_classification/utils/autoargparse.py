from enum import Enum
import os as _os
import sys as _sys
from typing import Any

from box import Box


class ParamType(Enum):
    """Enum to represent the type of a parameter."""

    KEY = 0
    VALUE = 1


class ArgumentParser(object):
    """ArgsManager is a class that manages command line arguments."""

    def __init__(
        self,
        deep_parse: bool = True,
        list_append: bool = False,
    ):
        """Initializes the ArgsManager.

        Args:
            deep_parse (bool): If True, the parser will attempt to parse nested arguments. Defaults to True.
            list_append (bool): If True, list values will be appended instead of replaced. Defaults to False.
        """
        self.deep_parse = deep_parse
        self.list_append = list_append

    def parse_args(self, namespace: Box | None = None) -> Box:
        # Args are redirected to the system arguments
        args = _sys.argv[1:]

        # If namespace is not provided, create a new Namespace to store args
        if namespace is None:
            namespace = Box()

        # parse the arguments
        while args:
            param = args.pop(0)
            assert self._judge_param_type(param), f"Expected a key, but got: {param}"

            # Create a value list to store values for the current key
            value_list = []
            while args and self._judge_param_type(args[0]) == ParamType.VALUE:
                value = args.pop(0)
                is_number, number = self._to_number(value)
                if is_number:
                    value_list.append(number)
                else:
                    value_list.append(value)

            # If no values were provided, set the value to True
            if not value_list:
                value = True
            elif len(value_list) == 1:
                value = value_list[0]
            else:
                value = value_list

            self._add_to_namespace(namespace, param[2:], value)

        return namespace

    def parse_known_args(self, names: list[str], args: list[str], namespace: Box | None = None):
        raise NotImplementedError

    def _judge_param_type(self, param: str) -> ParamType:
        """Checks if the next parameter is a value or a key.

        Args:
            param (str): The parameter string, e.g., '--key' '2e-3' 'value'.

        Returns:
            A tuple containing a boolean indicating if the parameter has a value,
            and the key without the leading '--'.
        """
        if not param.startswith('--'):
            return ParamType.VALUE

        return ParamType.KEY

    def _to_number(self, param: str) -> tuple[bool, int | float | complex | None]:
        """Converts a string to a digit if possible, otherwise returns None.

        .. note:: This method does not support non-decimal numbers.

        Args:
            param (str): The parameter string to convert, e.g., '2e-3', '3.14', '1+2j'.

        Returns:
            The converted digit if successful, otherwise None.
        """
        try:
            num = complex(param)
            if num.imag == 0:
                if num.real.is_integer():
                    return True, int(num.real)
                return True, float(num.real)
            return True, num
        except Exception:
            return False, None

    def _add_to_namespace(self, namespace: Box, key: str, value: Any) -> None:
        """Adds a key-value pair to the namespace.

        Args:
            namespace (Box): The Namespace object to which the key-value pair will be added.
            key (str): The key to add.
            value (Any): The value to associate with the key.
        """
        # print(f"Adding to namespace: {key} = {value}")
        link_keys = key.split('.')
        if self.deep_parse and len(link_keys) > 1:
            for link_key in link_keys[:-1]:
                # Assure the key is a valid identifier
                assert link_key.isidentifier(), f"Invalid key: '{link_key}'. Keys must be valid identifiers."

                # If the namespace has the key, navigate to it
                if link_key in namespace:
                    # If the key already exists, navigate to it
                    namespace = namespace[link_key]
                else:
                    # If the key does not exist, create a new Namespace
                    namespace[link_key] = Box()
                    namespace = namespace[link_key]
            # After navigating through the keys, set the value
            self._add_to_namespace(namespace, link_keys[-1], value)
        else:
            # Ensure the key is a valid identifier
            assert key.isidentifier(), f"Invalid key: '{key}'. Keys must be valid identifiers."

            # If the key does not exist in the namespace, add it
            if key not in namespace:
                namespace[key] = value
            else:
                existing_value = namespace[key]
                if isinstance(existing_value, list):
                    if not self.list_append:
                        if isinstance(value, list):
                            # If the existing value is a list and list_append is True, replace it
                            namespace[key] = value
                        else:
                            namespace[key] = [value]
                    else:
                        # If the existing value is a list, append the new value
                        if isinstance(value, list):
                            existing_value.extend(value)
                        else:
                            existing_value.append(value)
                else:
                    namespace[key] = value


def deep_update(raw_space: Box, new_space: Box) -> None:
    """Recursively updates a namespace with another namespace.

    Args:
        raw_space (Box): The namespace to update, which is also the result receiver.
        new_space (Box): The namespace with new values to update.
    """
    for k, v in new_space.items():
        if k in raw_space and isinstance(raw_space[k], Box) and isinstance(v, Box):
            deep_update(raw_space[k], v)
        else:
            setattr(raw_space, k, v)


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     args = parser.parse_args()
#     print(args)
