# coding=utf-8
# Copyright (c) 2016, Cedric Zhuang
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of disclaimer nor the names of its contributors may
#       be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from setuptools import setup
import io
import re
import os

__author__ = 'Cedric Zhuang'


def version():
    desc = get_long_description()
    ret = re.findall(r'VERSION: (.*)', desc)[0]
    return ret.strip()


def here(filename=None):
    ret = os.path.abspath(os.path.dirname(__file__))
    if filename is not None:
        ret = os.path.join(ret, filename)
    return ret


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n\n')
    buf = []
    for filename in filenames:
        with io.open(here(filename), encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


def read_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


def get_long_description():
    filename = 'README.rst'
    return read(filename)


setup(
    name="stockstats",
    version=version(),
    author="Cedric Zhuang",
    author_email="jealous@163.com",
    description="DataFrame with inline stock statistics support.",
    license="BSD",
    keywords="stock statistics indicator",
    url="http://github.com/jealous/stockstats",
    py_modules=['stockstats'],
    platforms=['any'],
    long_description=get_long_description(),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: Implementation :: CPython",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=read_requirements('requirements.txt'),
    tests_require=read_requirements('test-requirements.txt')
)
