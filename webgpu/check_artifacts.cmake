foreach(extension IN ITEMS html js wasm)
  set(path "${ARTIFACT_BASE}.${extension}")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "missing WebGPU artifact: ${path}")
  endif()
  file(SIZE "${path}" size)
  if(size EQUAL 0)
    message(FATAL_ERROR "empty WebGPU artifact: ${path}")
  endif()
endforeach()

get_filename_component(artifact_dir "${ARTIFACT_BASE}" DIRECTORY)
foreach(asset IN ITEMS browser_ui.js verification_worker.js residual_plot.js free_boundary.js coil_geometry.js equilibrium_view.js orbit_renderer.js boundary_editor.js cumes_coils.mjs cumes_coils.wasm)
  if(NOT EXISTS "${artifact_dir}/${asset}")
    message(FATAL_ERROR "missing WebGPU frontend: ${asset}")
  endif()
endforeach()

file(READ "${ARTIFACT_BASE}.html" html)
if(NOT html MATCHES "src=\\\"cumes_webgpu\\.js\\?v=[0-9a-f]+\\\"")
  message(FATAL_ERROR
    "WebGPU HTML does not use a content-versioned JavaScript URL")
endif()
foreach(marker IN ITEMS
    "Input boundary"
    "Boundary editor mode"
    "R cosine coefficients"
    "Z sine coefficients"
    "RMS fit error"
    "Run equilibrium"
    "Flux surfaces"
    "Interactive three-dimensional equilibrium")
  string(FIND "${html}" "${marker}" marker_offset)
  if(marker_offset EQUAL -1)
    message(FATAL_ERROR
      "WebGPU HTML is missing interactive-app marker: ${marker}")
  endif()
endforeach()

file(READ "${ARTIFACT_BASE}.js" javascript)
foreach(marker IN ITEMS "publish_browser_equilibrium" "requested_app_mode"
    "cumesIterationTiming" "Device compute (GPU)" "Host non-wait elapsed")
  string(FIND "${javascript}" "${marker}" marker_offset)
  if(marker_offset EQUAL -1)
    message(FATAL_ERROR
      "WebGPU JavaScript is missing browser bridge: ${marker}")
  endif()
endforeach()

get_filename_component(artifact_dir "${ARTIFACT_BASE}" DIRECTORY)
foreach(preset IN ITEMS solovev w7x cth_like)
  foreach(asset IN ITEMS "${preset}.json" "coils.${preset}")
    if(NOT EXISTS "${artifact_dir}/presets/${asset}")
      message(FATAL_ERROR "missing coil preset: ${asset}")
    endif()
  endforeach()
endforeach()
file(GLOB_RECURSE field_grids "${artifact_dir}/presets/*.nc")
if(field_grids)
  message(FATAL_ERROR "Browser presets must generate field grids at runtime")
endif()
