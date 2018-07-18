# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Utility for saving and loading configuration from TOML files, since we have
a lot of knobs we can turn.

"""


def merge_nested_dicts(base, more):
    """Given two dictionaries that may contain sub-dictionarie, merge *more* into
    *base*, overwriting duplicated values.

    """
    for key, val in more.items():
        if isinstance(val, dict):
            base_val = base.setdefault(key, {})

            if not isinstance(base_val, dict):
                raise Exception('trying to merge a dictionary named "%s" into a non-dictionary %r'
                                % (key, base_val))

            merge_nested_dicts(base_val, val)
        else:
            base[key] = val

    return base


def load_tomls_with_inheritance(path):
    import os.path
    import pytoml

    to_load = [path]
    dicts = []

    while len(to_load):
        this_path = to_load[0]

        with open(this_path, 'rt') as f:
            this_dict = pytoml.load(f)

        inherit_spec = this_dict.get('inherit')

        if inherit_spec is None:
            to_inherit = []
        elif isinstance(inherit_spec, str):
            to_inherit = [inherit_spec]
        elif isinstance(inherit_spec, list):
            to_inherit = inherit_spec
        else:
            raise Exception('unhandled \"inherit\" specification in config file \"%s\": %r'
                            % (this_path, inherit_spec))

        for item in to_inherit:
            to_load.append(os.path.join(os.path.dirname(this_path), item))

        dicts.append(this_dict)
        to_load = to_load[1:]

    data = {}

    for item in dicts[::-1]:
        merge_nested_dicts(data, item)

    return data


class Configuration(object):
    """A simple class for saving and loading configuration from TOML files.

    Configuration values should be declared as class properties with default
    values specified.

    If the value of a class property is a type that is a subclass of
    Configuration, that value will be filled in by creating an instance of
    that type and reading in its values in the same way.

    """
    def __init__(self):
        # If there are any sub config items, we need to initialize them to
        # defaults too.

        for name, default in self.__config_items():
            if isinstance(default, type) and issubclass(default, Configuration):
                setattr(self, name, default())


    @classmethod
    def __config_items(cls):
        for name, default in cls.__dict__.items():
            if name[0] != '_' and (isinstance(default, type) or not callable(default)):
                yield name, default


    @classmethod
    def from_collection(cls, obj):
        inst = cls()
        my_section = obj.get(cls.__section__, {})

        if not isinstance(my_section, dict):
            raise RuntimeError('configuration item "%s" should be a dict, but is instead %r'
                               % (cls.__section__, my_section))

        for name, default in cls.__config_items():
            if isinstance(default, type) and issubclass(default, Configuration):
                setattr(inst, name, default.from_collection(obj))
            elif name in my_section:
                setattr(inst, name, my_section[name])

        return inst


    def to_collection(self, obj):
        my_section = obj.setdefault(self.__class__.__section__, {})

        for name, default in self.__config_items():
            if isinstance(default, type) and issubclass(default, Configuration):
                getattr(self, name).to_collection(obj)
            else:
                my_section[name] = getattr(self, name)

        return self


    @classmethod
    def from_toml(cls, path):
        data = load_tomls_with_inheritance(path)

        try:
            return cls.from_collection(data)
        except Exception as e:
            raise Exception('error loading configuration from file "%s"' % path) from e


    @classmethod
    def update_toml(cls, path):
        """Update a config file, preserving existing known entries but adding values
        for parameters that weren't given values explicitly before.

        Note that this intentionally does not use the inheritance scheme,
        since we don't want to add all of the inherited values to the existing
        file.

        """
        import pytoml

        try:
            with open(path, 'rt') as f:
                data = pytoml.load(f)
        except FileNotFoundError: # yay Python 3!
            data = {}

        inst = cls.from_collection(data)
        inst.to_collection(data)

        with open(path, 'wt') as f:
            pytoml.dump(f, data, sort_keys=True)


    @classmethod
    def generate_config_cli(cls, prog_name, args):
        from argparse import ArgumentParser
        ap = ArgumentParser(prog=prog_name)
        ap.add_argument('config_path', metavar='CONFIG-PATH',
                        help='The path of the config file to create or update.')
        settings = ap.parse_args(args=args)
        cls.update_toml(settings.config_path)
