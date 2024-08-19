#!/bin/bash
# fix the downloading issues before
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1jF2weMxWEi4AolGW1OT73wRsudI3i3zZ" -O "data.zip"
unzip data.zip
rm data.zip