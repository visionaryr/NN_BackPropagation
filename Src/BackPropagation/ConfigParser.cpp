/**
  Configuration Parser implementation.

  Copyright (c) 2026, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.

**/

#include "ConfigParser.h"

#include <fstream>
#include <sstream>
#include <functional>

using namespace std;

/**
  Constructor for ConfigParser class.
**/
ConfigParser::ConfigParser (
  void
  )
{
  // Initialize section parsers map
  SectionParsers[SECTION_NAME_NETWORK] = &ConfigParser::NetworkConfigParser;
  SectionParsers[SECTION_NAME_TRAINING] = &ConfigParser::TrainingConfigParser;
  SectionParsers[SECTION_NAME_DATA] = &ConfigParser::DataConfigParser;
}

/**
  Trim whitespace from both ends of a string.

  @param[in]  Str   The string to trim.

  @return     Trimmed string.

**/
string
ConfigParser::Trim (
  const string& Str
  )
{
  size_t First = Str.find_first_not_of(" \t\r\n");
  if (First == string::npos) {
    return "";
  }
  size_t Last = Str.find_last_not_of(" \t\r\n");
  return Str.substr(First, (Last - First + 1));
}

/**
  Parse a line containing key: value.

  @param[in]  Line        The line to parse.
  @param[out] Key         Extracted key.
  @param[out] Value       Extracted value.

  @return     True if line was successfully parsed, false otherwise.
**/
bool
ConfigParser::ParseKeyValue (
  const string& Line,
  string& Key,
  string& Value
  )
{
  size_t ColonPos = Line.find(':');
  if (ColonPos == string::npos) {
    return false;
  }

  Key = Trim(Line.substr(0, ColonPos));
  Value = Trim(Line.substr(ColonPos + 1));

  // Ignore empty keys or comments
  if (Key.empty() || Key[0] == '#') {
    return false;
  }

  return true;
}

/**
  Parse a list in YAML format: [val1, val2, val3].

  @param[in]  ListStr    The list string to parse.

  @return     Vector of parsed values as strings.

**/
vector<string>
ConfigParser::ParseList (
  const string& ListStr
  )
{
  vector<string> Result;

  // Remove brackets
  string Content = ListStr;
  if (!Content.empty() && Content[0] == '[') {
    Content = Content.substr(1);
  }
  if (!Content.empty() && Content[Content.length() - 1] == ']') {
    Content = Content.substr(0, Content.length() - 1);
  }

  // Split by comma
  stringstream Ss(Content);
  string Item;
  while (getline(Ss, Item, ',')) {
    string TrimmedItem = Trim(Item);
    if (!TrimmedItem.empty()) {
      Result.push_back(TrimmedItem);
    }
  }

  return Result;
}

/**
  Convert string value to unsigned int.

  @param[in]  Value   The string value to convert.

  @return     Converted unsigned int value.
  
  @throw      std::invalid_argument if conversion fails.

**/
unsigned int
ConfigParser::StringToUint (
  const string& Value
  )
{
  try {
    return stoul(Value);
  } catch (const exception& e) {
    throw invalid_argument("Cannot convert '" + Value + "' to unsigned int");
  }
}

/**
  Convert string value to double.

  @param[in]  Value   The string value to convert.

  @return     Converted double value.
  
  @throw      std::invalid_argument if conversion fails.

**/
double
ConfigParser::StringToDouble (
  const string& Value
  )
{
  try {
    return stod(Value);
  } catch (const exception& e) {
    throw invalid_argument("Cannot convert '" + Value + "' to double");
  }
}

/**
  Parser to handle key-value pair in network configuration section.

  @param[in]  Key     The configuration key.
  @param[in]  Value   The configuration value.
  @param[out] Config  The configuration data structure to update. 

**/
void
ConfigParser::NetworkConfigParser (
  const string&     Key,
  const string&     Value,
  ConfigurationData &Config
  )
{
  if (Key == KEY_NAME_LAYOUT) {
    vector<string> LayoutStr = ParseList(Value);
    for (const auto& Item : LayoutStr) {
      Config.Network.Layout.push_back(StringToUint(Item));
    }
  }
}

/**
  Parser to handle key-value pair in training configuration section.

  @param[in]  Key     The configuration key.
  @param[in]  Value   The configuration value.
  @param[out] Config  The configuration data structure to update. 

**/
void
ConfigParser::TrainingConfigParser (
  const string&     Key,
  const string&     Value,
  ConfigurationData &Config
  )
{
  string TrainingMode;

  if (Key == KEY_NAME_LEARNING_RATE) {
    Config.Training.LearningRate = StringToDouble(Value);
  } else if (Key == KEY_NAME_EPOCHS) {
    Config.Training.Epochs = StringToUint(Value);
  } else if (Key == KEY_NAME_TARGET_LOSS) {
    Config.Training.TargetLoss = StringToDouble(Value);
  } else if (Key == KEY_NAME_BATCH_SIZE) {
    Config.Training.BatchSize = StringToUint(Value);
  } else if (Key == KEY_NAME_TRAINING_MODE) {
    // Remove quotes if present
    if ((Value[0] == '"' && Value[Value.length() - 1] == '"') ||
        (Value[0] == '\'' && Value[Value.length() - 1] == '\'')) {
      TrainingMode = Value.substr(1, Value.length() - 2);
    }
    
    if (TrainingMode == "BATCH_MODE") {
      Config.Training.TrainingMode = BATCH_MODE;
    } else if (TrainingMode == "PATTERN_MODE") {
      Config.Training.TrainingMode = PATTERN_MODE;
    } else {
      throw invalid_argument("Invalid training mode: " + TrainingMode);
    }
  }
}

/**
  Parser to handle key-value pair in data configuration section.

  @param[in]  Key     The configuration key.
  @param[in]  Value   The configuration value.
  @param[out] Config  The configuration data structure to update. 

**/
void
ConfigParser::DataConfigParser (
  const string&     Key,
  const string&     Value,
  ConfigurationData &Config
  )
{
  if (Key == KEY_NAME_TRAINING_CATEGORIES) {
    vector<string> CategoriesStr = ParseList(Value);
    for (const auto& Item : CategoriesStr) {
      Config.Data.TrainingCategories.push_back(StringToUint(Item));
    }
  }
}

/**
  Parse YAML configuration file and return AppConfig structure.

  @param[in]  ConfigFilePath   Path to the YAML configuration file.

  @return     AppConfig structure with all parsed configuration values.
  
  @throw      std::runtime_error if file cannot be opened or parsed.

**/
ConfigurationData
ConfigParser::ParseConfigFile (
  const string& ConfigFilePath
  )
{
  ifstream ConfigFile(ConfigFilePath);
  if (!ConfigFile.is_open()) {
    throw runtime_error("Cannot open configuration file: " + ConfigFilePath);
  }

  ConfigurationData Config;
  string Line;
  string CurrentSection;

  while (getline(ConfigFile, Line)) {
    // Skip empty lines and comments
    string TrimmedLine = Trim(Line);
    if (TrimmedLine.empty() || TrimmedLine[0] == '#') {
      continue;
    }

    // Detect section headers (format: section:)
    if (TrimmedLine[TrimmedLine.length() - 1] == ':' && TrimmedLine.find(' ') == string::npos) {
      CurrentSection = TrimmedLine.substr(0, TrimmedLine.length() - 1);
      continue;
    }

    string Key, Value;
    if (!ParseKeyValue(TrimmedLine, Key, Value)) {
      continue;
    }

    // Parse based on current section
    if (SectionParsers.find(CurrentSection) != SectionParsers.end()) {
      SectionParsers[CurrentSection](Key, Value, Config);
    }
  }

  ConfigFile.close();

  Config.SetValid ();

  return Config;
}
