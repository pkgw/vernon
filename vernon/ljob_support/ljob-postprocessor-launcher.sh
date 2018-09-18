#! /bin/bash
if [ -z "$IPSWICH_ROOT" ] ; then
    echo >&2 "error: no enviroment variable \$IPSWICH_ROOT; improperly launched?"
    exit 1
fi

export PATH="$IPSWICH_ROOT/env/bin:$PATH"
exec python postprocess.py
