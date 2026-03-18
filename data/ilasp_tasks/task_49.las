#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o3).
#constant(obj, o4).
#constant(obj, o6).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d2(var(obj))).
#modeh(d3(var(obj))).
#modeh(d4(var(obj))).
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
  d2(o0), d2(o9), d3(o0), d3(o9), d4(o0), d4(o9), d5(o0), d5(o9)
}, {
  d2(o1), d2(o4), d2(o6), d2(o8), d3(o1), d3(o3), d3(o4), d3(o6), d3(o8), d4(o1), d4(o3), d4(o4), d4(o8), d5(o1), d5(o3), d5(o4), d5(o6), d5(o8)
}, {
  d0(o0).
  d0(o1).
  d0(o4).
  d0(o8).
  d0(o9).
  d1(o0).
  d1(o9).
  d2(o3).
  d4(o6).
}).