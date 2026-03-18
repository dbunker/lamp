#constant(obj, o0).
#constant(obj, o2).
#constant(obj, o7).
#constant(obj, o9).

#modeh(d0(var(obj))).
#modeh(d1(var(obj))).

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
  d0(o0), d1(o0)
}, {
  d0(o7), d1(o2), d1(o7), d1(o9)
}, {
  d0(o2).
  d0(o9).
  d2(o0).
  d3(o0).
  d4(o0).
  d5(o7).
}).