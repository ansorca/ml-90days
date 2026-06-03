

def word_frequency(text)
  result = {}
  return result if text == nil || text.length == 0
  text.split.each do
    result[it] ||= 0
    result[it] += 1
  end
  result
end
puts word_frequency("one two three four one two")
