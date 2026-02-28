/**
  Configuration Parser for YAML configuration files.

  Copyright (c) 2026, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.

**/

#ifndef _CONFIG_PARSER_H_
#define _CONFIG_PARSER_H_

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <functional>

#include "ConfigurationData.h"

#define SECTION_NAME_NETWORK   "Network"
#define SECTION_NAME_TRAINING  "Training"
#define SECTION_NAME_DATA      "Data"

#define KEY_NAME_LAYOUT                 "Layout"
#define KEY_NAME_LEARNING_RATE          "LearningRate"
#define KEY_NAME_EPOCHS                 "Epochs"
#define KEY_NAME_TARGET_LOSS            "TargetLoss"
#define KEY_NAME_BATCH_SIZE             "BatchSize"
#define KEY_NAME_TRAINING_MODE          "TrainingMode"
#define KEY_NAME_TRAINING_CATEGORIES    "TrainingCategories"

class ConfigParser {
  public:
    /**
      Constructor for ConfigParser class.
    **/
    ConfigParser (
      void
      );

    /**
      Parse YAML configuration file and return AppConfig structure.

      @param[in]  ConfigFilePath   Path to the YAML configuration file.

      @return     AppConfig structure with all parsed configuration values.
      
      @throw      std::runtime_error if file cannot be opened or parsed.
    **/
    ConfigurationData ParseConfigFile(const std::string& ConfigFilePath);

  private:
    /**
      Trim whitespace from both ends of a string.

      @param[in]  Str   The string to trim.

      @return     Trimmed string.
    **/
    static std::string Trim(const std::string& Str);

    /**
      Parse a line containing key: value.

      @param[in]  Line        The line to parse.
      @param[out] Key         Extracted key.
      @param[out] Value       Extracted value.

      @return     True if line was successfully parsed, false otherwise.
    **/
    static bool ParseKeyValue(const std::string& Line, std::string& Key, std::string& Value);

    /**
      Parse a list in YAML format: [val1, val2, val3].

      @param[in]  ListStr    The list string to parse.

      @return     Vector of parsed values as strings.
    **/
    static std::vector<std::string> ParseList(const std::string& ListStr);

    /**
      Convert string value to unsigned int.

      @param[in]  Value   The string value to convert.

      @return     Converted unsigned int value.
      
      @throw      std::invalid_argument if conversion fails.
    **/
    static unsigned int StringToUint(const std::string& Value);

    /**
      Convert string value to double.

      @param[in]  Value   The string value to convert.

      @return     Converted double value.
      
      @throw      std::invalid_argument if conversion fails.
    **/
    static double StringToDouble(const std::string& Value);

    /**
      Parser to handle key-value pair in network configuration section.

      @param[in]  Key     The configuration key.
      @param[in]  Value   The configuration value.
      @param[out] Config  The configuration data structure to update. 

    **/
    static
    void
    NetworkConfigParser (
      const std::string&     Key,
      const std::string&     Value,
      ConfigurationData &Config
      );
    /**
      Parser to handle key-value pair in training configuration section.

      @param[in]  Key     The configuration key.
      @param[in]  Value   The configuration value.
      @param[out] Config  The configuration data structure to update. 

    **/
    static
    void
    TrainingConfigParser (
      const std::string&     Key,
      const std::string&     Value,
      ConfigurationData &Config
      );

    /**
      Parser to handle key-value pair in data configuration section.

      @param[in]  Key     The configuration key.
      @param[in]  Value   The configuration value.
      @param[out] Config  The configuration data structure to update. 

    **/
    static
    void
    DataConfigParser (
      const std::string&     Key,
      const std::string&     Value,
      ConfigurationData &Config
      );

    std::map<std::string, std::function<void(const std::string&, const std::string&, ConfigurationData&)>> SectionParsers;
};

#endif
