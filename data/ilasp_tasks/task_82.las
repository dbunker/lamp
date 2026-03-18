#constant(obj, o2).
#constant(obj, o6).
#constant(obj, o7).

#modeh(d1(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d1(o2), d1(o7), d3(o7), d4(o2)
}, {
  d1(o6), d3(o6)
}, {
  d3(o2).
  d4(o6).
  d4(o7).
  d5(o2).
  d5(o7).
}).