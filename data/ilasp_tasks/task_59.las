#constant(obj, o2).
#constant(obj, o3).
#constant(obj, o5).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d0(var(obj))).

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

#pos(eg1, {
  d0(o8)
}, {
  d0(o2), d0(o3), d0(o5)
}, {
  d0(o9).
  d1(o3).
  d1(o8).
  d2(o8).
  d3(o5).
  d4(o2).
}).