#constant(obj, o1).
#constant(obj, o3).
#constant(obj, o6).
#constant(obj, o7).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d5(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d5(o3)
}, {
  d5(o1), d5(o6), d5(o7), d5(o8), d5(o9)
}, {
  d0(o3).
  d1(o3).
  d1(o6).
  d1(o8).
  d2(o7).
  d3(o3).
  d3(o9).
  d4(o1).
  d4(o3).
}).