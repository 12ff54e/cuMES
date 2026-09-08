if(NOT DEFINED ARTIFACT_BASE)
  message(FATAL_ERROR "ARTIFACT_BASE is required")
endif()

set(html "${ARTIFACT_BASE}.html")
set(javascript "${ARTIFACT_BASE}.js")
set(wasm "${ARTIFACT_BASE}.wasm")
foreach(artifact IN ITEMS "${html}" "${javascript}" "${wasm}")
  if(NOT EXISTS "${artifact}")
    message(FATAL_ERROR "missing WebGPU artifact: ${artifact}")
  endif()
endforeach()

file(SHA256 "${javascript}" javascript_hash)
file(SHA256 "${wasm}" wasm_hash)
file(SHA256 "${FRONTEND_SOURCE_DIR}/verification_worker.js" worker_hash)
string(SHA256 asset_hash "${javascript_hash}${wasm_hash}${worker_hash}")
string(SUBSTRING "${asset_hash}" 0 16 asset_version)

file(READ "${html}" contents)
get_filename_component(artifact_dir "${ARTIFACT_BASE}" DIRECTORY)
foreach(asset IN ITEMS browser_ui.js verification_worker.js residual_plot.js free_boundary.js equilibrium_view.js boundary_editor.js)
  configure_file("${FRONTEND_SOURCE_DIR}/${asset}" "${artifact_dir}/${asset}" COPYONLY)
  file(SHA256 "${artifact_dir}/${asset}" frontend_hash)
  string(REPLACE "src=\"${asset}\"" "src=\"${asset}?v=${frontend_hash}\"" contents "${contents}")
  string(REPLACE "src=${asset}" "src=\"${asset}?v=${frontend_hash}\"" contents "${contents}")
endforeach()
string(FIND "${contents}" "src=cumes_webgpu.js" unquoted_position)
string(FIND "${contents}" "src=\"cumes_webgpu.js\"" quoted_position)
if(unquoted_position EQUAL -1 AND quoted_position EQUAL -1)
  message(FATAL_ERROR "generated WebGPU runtime script tag was not found")
endif()
string(REPLACE "src=cumes_webgpu.js"
               "src=\"cumes_webgpu.js?v=${asset_version}\""
               contents "${contents}")
string(REPLACE "src=\"cumes_webgpu.js\""
               "src=\"cumes_webgpu.js?v=${asset_version}\""
               contents "${contents}")
file(WRITE "${html}" "${contents}")

file(COPY "${FRONTEND_SOURCE_DIR}/presets/" DESTINATION "${artifact_dir}/presets")
foreach(preset IN ITEMS solovev cth_like)
  configure_file("${FRONTEND_SOURCE_DIR}/../deps/vacuum-field/tests/data/coils.${preset}"
                 "${artifact_dir}/presets/coils.${preset}" COPYONLY)
endforeach()

configure_file("${FRONTEND_SOURCE_DIR}/../inputs/w7x.json"
               "${artifact_dir}/presets/fixed-w7x.json" COPYONLY)
