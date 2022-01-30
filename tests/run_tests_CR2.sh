#!/bin/bash
## only used for converting obj model files
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export PYTHONPATH=$DIR/../:$PYTHONPATH

usage()
{
    echo "usage: ./run_tests.sh [[-c] | [-py3] | [-h]]"
}

coverage=0
python_version=2
while [ "$1" != "" ]; do
    case $1 in
        -c | --coverage )           coverage=1
                                    ;;
        -py3 | --python-version-3 ) python_version=3
                                    ;;
        -h | --help )               usage
                                    exit
                                    ;;
        * )                         usage
                                    exit 1
    esac
    shift
done

PYV=`python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";`
# if the default Python is v3.x then we force using the -py3 switch
if (( $(echo "$PYV > 3.0" | bc -l) )); then
    python_version=3
fi

# Convert SUNCG test data
$DIR/../scripts/convert_suncg.sh "$DIR/data/RealRoom/room/CR2"

