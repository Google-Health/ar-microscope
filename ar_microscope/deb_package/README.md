# ARM Configuration

The ARM is configured by two settings files:

  1. (mandatory) `/usr/local/share/arm_configs/default_config.textproto`
  2. (optional) `/usr/local/etc/arm_custom_config.textproto`

The default settings are those in `default_config`. These settings are
overridden by the settings in `arm_custom_config`. The `default_config` file is
replaced with the latest version upon ARM software updates, while the `arm_custom_config` is untouched.

The file `/usr/local/share/arm_configs/default_config.textproto` provides a
template for the settings that should be configured per ARM. This file will be
copied to `arm_custom_config` upon installation, if `arm_custom_config` does not
already exist.


## Configuring ARM

The config contains configuration for ARM's models, microdisplay, and
objective positions. These can be manually overriden as needed.
