#constant(obj, o1).
#constant(obj, o2).

#modeh(d2(var(obj))).

#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).

#pos(eg1, {
  d2(o1)
}, {
  d2(o2)
}, {
  d1(o1).
  d3(o2).
  d4(o1).
  d4(o2).
}).