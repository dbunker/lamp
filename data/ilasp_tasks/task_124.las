#constant(obj, o0).
#constant(obj, o3).
#constant(obj, o4).

#modeh(d4(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).

#pos(eg1, {
  d4(o4)
}, {
  d4(o3)
}, {
  d0(o3).
  d2(o4).
  d3(o4).
  d4(o0).
}).