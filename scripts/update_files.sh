# Used to replace lines in the raijin job scripts

#!/bin/bash
for AGE in {0..12}
do
    # sed -i -e 's/\/short\/kc5\///g' fit_twa$AGE.txt
    # sed -i -e 's/TWA_traceback_20Myr/TWA_core_traceback_15Myr/g' fit_twa$AGE.txt
    sed -i -e 's/walltime=.*/walltime=00:20:00/g' fit_twa$AGE.txt
done
