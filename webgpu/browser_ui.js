// UI-only helpers. The numerical runtime may live on the page or in a worker.
function createCumesLogBuffer(output, schedule = setTimeout, cancel = clearTimeout) {
  let pending = [], timer = null;
  const text = output.ownerDocument.createTextNode('');
  output.append(text);
  function flush() {
    if (timer !== null) cancel(timer);
    timer = null;
    if (!pending.length) return;
    text.appendData(pending.join(''));
    pending = [];
    output.scrollTop = output.scrollHeight;
  }
  return {
    append(line) {
      pending.push(line + '\n');
      if (timer === null) timer = schedule(flush, 100);
    },
    flush
  };
}
