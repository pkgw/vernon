# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Utility for saving and loading configuration from TOML files, since we have
a lot of knobs we can turn.

"""


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
            if name[0] != '_' and not callable(default):
                yield name, default


    @classmethod
    def from_collection(cls, obj):
        inst = cls()
        my_section = obj.get(cls.__section__)

        if my_section is None:
            raise RuntimeError('missing required configuration section "%s"' % cls.__section__)
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
        import pytoml

        with open(path, 'rt') as f:
            data = pytoml.load(f)

        try:
            return cls.from_collection(data)
        except Exception as e:
            raise Exception('error loading configuration from file "%s"' % path) from e


    @classmethod
    def update_toml(cls, path):
        """Update a config file, preserving existing known entries but adding values
        for parameters that weren't given values explicitly before.

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
