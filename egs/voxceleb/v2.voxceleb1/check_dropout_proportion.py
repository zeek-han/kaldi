#!/usr/bin/env python3

_debug_dropout = False

def _parse_dropout_option(dropout_option):
    """Parses the string option to --trainer.dropout-schedule and
    returns a list of dropout schedules for different component name patterns.
    Calls _parse_dropout_string() function for each component name pattern
    in the option.

    Arguments:
        dropout_option: The string option passed to --trainer.dropout-schedule.
            See its help for details.
            See _self_test() for examples.
        num_archive_to_process: See _parse_dropout_string() for details.

    Returns a list of (component_name, dropout_schedule) tuples,
    where dropout_schedule is itself a list of
    (data_fraction, dropout_proportion) tuples sorted in reverse order of
    data_fraction.
    A data fraction of 0 corresponds to beginning of training
    and 1 corresponds to all data.
    """
    components = dropout_option.strip().split(' ')
    dropout_schedule = []
    for component in components:
        parts = component.split('=')

        if len(parts) == 2:
            component_name = parts[0]
            this_dropout_str = parts[1]
        elif len(parts) == 1:
            component_name = '*' 
            this_dropout_str = parts[0]
        else:
            raise Exception("The dropout schedule must be specified in the "
                            "format 'pattern1=func1 patter2=func2' where "
                            "the pattern can be omitted for a global function "
                            "for all components.\n"
                            "Got {0} in {1}".format(component, dropout_option))

        this_dropout_values = _parse_dropout_string(this_dropout_str)
        dropout_schedule.append((component_name, this_dropout_values))

    if _debug_dropout:
        logger.info("Dropout schedules for component names is as follows:")
        logger.info("<component-name-pattern>: [(num_archives_processed), "
                    "(dropout_proportion) ...]")
        for name, schedule in dropout_schedule:
            logger.info("{0}: {1}".format(name, schedule))

    return dropout_schedule

def _parse_dropout_string(dropout_str):
    """Parses the dropout schedule from the string corresponding to a
    single component in --trainer.dropout-schedule.
    This is a module-internal function called by parse_dropout_function().

    Arguments:
        dropout_str: Specifies dropout schedule for a particular component
            name pattern.
            See help for the option --trainer.dropout-schedule.

    Returns a list of (data_fraction_processed, dropout_proportion) tuples
    sorted in descending order of num_archives_processed.
    A data fraction of 1 corresponds to all data.
    """
    dropout_values = []
    parts = dropout_str.strip().split(',')

    try:
        if len(parts) < 2:
            raise Exception("dropout proportion string must specify "
                            "at least the start and end dropouts")

        # Starting dropout proportion
        dropout_values.append((0, float(parts[0])))
        for i in range(1, len(parts) - 1): 
            value_x_pair = parts[i].split('@')
            if len(value_x_pair) == 1:
                # Dropout proportion at half of training
                dropout_proportion = float(value_x_pair[0])
                data_fraction = 0.5 
            else:
                assert len(value_x_pair) == 2

                dropout_proportion = float(value_x_pair[0])
                data_fraction = float(value_x_pair[1])

            if (data_fraction < dropout_values[-1][0]
                    or data_fraction > 1.0):
                logger.error(
                    "Failed while parsing value %s in dropout-schedule. "
                    "dropout-schedule must be in incresing "
                    "order of data fractions.", value_x_pair)
                raise ValueError

            dropout_values.append((data_fraction, float(dropout_proportion)))

        dropout_values.append((1.0, float(parts[-1])))
    except Exception:
        logger.error("Unable to parse dropout proportion string %s. "
                     "See help for option "
                     "--trainer.dropout-schedule.", dropout_str)
        raise

    # reverse sort so that its easy to retrieve the dropout proportion
    # for a particular data fraction
    dropout_values.reverse()
    for data_fraction, proportion in dropout_values:
        assert data_fraction <= 1.0 and data_fraction >= 0.0
        assert proportion <= 1.0 and proportion >= 0.0

    return dropout_values

if __name__ == '__main__':
    aa = _parse_dropout_option('0,0@0.20,0.1@0.50,0')
    print(type(aa[0]))
    print(len(aa[0]))
    a, b = aa[0]
    print(type(a), a)
    print(type(b), len(b))
    for ii in range(len(b)):
        print(b[ii])

