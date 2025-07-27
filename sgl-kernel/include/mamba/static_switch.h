#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...        - code to execute for true and false
#define BOOL_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                    \
    if (COND) {                            \
      constexpr bool CONST_NAME = true;    \
      return __VA_ARGS__();                \
    } else {                               \
      constexpr bool CONST_NAME = false;   \
      return __VA_ARGS__();                \
    }                                      \
  }()
