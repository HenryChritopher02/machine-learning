#!/bin/bash
git update-index --chmod=+x vina/vina_1.2.5_linux_x86_64
git commit -am "Mark Vina binary as executable"
git push
ls -l ./vina/vina_1.2.5_linux_x86_64  # Debug: Check permissions
