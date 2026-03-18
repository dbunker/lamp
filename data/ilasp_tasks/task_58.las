#constant(obj, o0).
#constant(obj, o4).
#constant(obj, o7).
#constant(obj, o9).

#modeh(d0(var(obj))).
#modeh(d2(var(obj))).
#modeh(d3(var(obj))).
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
  d0(o0), d2(o0), d3(o0), d5(o0)
}, {
  d0(o4), d0(o7), d0(o9), d2(o7), d2(o9), d3(o4), d3(o7), d3(o9), d5(o4), d5(o7), d5(o9)
}, {
  d1(o0).
  d1(o4).
  d1(o7).
  d2(o4).
  d4(o0).
  d4(o9).
}).