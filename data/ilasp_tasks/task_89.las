#constant(obj, o0).
#constant(obj, o5).
#constant(obj, o7).
#constant(obj, o8).

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
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d2(o5), d5(o5)
}, {
  d2(o7), d2(o8), d5(o0), d5(o7), d5(o8)
}, {
  d0(o7).
  d1(o5).
  d1(o8).
  d2(o0).
  d3(o5).
}).