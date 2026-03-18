#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o3).
#constant(obj, o6).
#constant(obj, o9).

#modeh(d0(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).
#modeh(d5(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d0(o1), d3(o9), d4(o0), d4(o3), d4(o6), d4(o9), d5(o1), d5(o3), d5(o9)
}, {
  d0(o6), d3(o0), d3(o1), d5(o6)
}, {
  d0(o0).
  d0(o3).
  d0(o9).
  d1(o3).
  d1(o9).
  d3(o3).
  d3(o6).
  d4(o1).
  d5(o0).
}).