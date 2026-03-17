#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o4).

#modeh(d0(var(obj))).
#modeh(d4(var(obj))).

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
  d0(o4), d4(o0)
}, {
  d0(o2), d4(o2), d4(o4)
}, {
  d0(o0).
  d1(o4).
  d2(o0).
  d3(o2).
}).