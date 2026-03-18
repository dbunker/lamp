#constant(obj, o0).
#constant(obj, o3).
#constant(obj, o4).

#modeh(d0(var(obj))).
#modeh(d1(var(obj))).
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
  d0(o4), d1(o3), d3(o3), d4(o0), d5(o0), d5(o3), d5(o4)
}, {
  d3(o0), d3(o4), d4(o3)
}, {
  d0(o0).
  d0(o3).
  d1(o0).
  d1(o4).
  d2(o3).
  d4(o4).
}).