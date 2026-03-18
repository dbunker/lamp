#constant(obj, o0).
#constant(obj, o1).
#constant(obj, o8).

#modeh(d0(var(obj))).
#modeh(d2(var(obj))).
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
  d0(o0), d2(o0), d5(o0)
}, {
  d0(o1), d0(o8), d2(o8), d5(o1), d5(o8)
}, {
  d1(o0).
  d2(o1).
  d3(o0).
  d3(o8).
  d4(o0).
}).