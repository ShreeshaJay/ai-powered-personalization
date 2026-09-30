# Commerce-Query Segmentation Rubric v2

This revision changes only the query-segmentation specificity axis. It adds a
plain-product-type class and defines a precedence order.

## Goal

- `known_item_navigation`: Reach a named product, model, title, person/author,
  brand storefront, retailer, website, or content destination.
- `product_discovery`: Find products satisfying a product type, attribute,
  recipient, occasion, or broad need.
- `comparison_decision`: Compare alternatives, seek recommendations, or decide
  what to buy.
- `informational_support`: Learn, troubleshoot, obtain instructions, or answer
  a product-related question.
- `service_account`: Reach an account, login, order, payment, delivery,
  subscription, repair, or customer-service workflow.
- `other_unclear`: Malformed, non-commerce, or insufficiently clear.

Goal rule: a query can be navigational even when the destination is outside
retail commerce. Commerce scope captures whether it belongs in the commerce
system.

## Object

- `product`
- `category`
- `brand_store`
- `service_content`
- `unclear`

Use `category` for a product type or merchandise family. Use `product` for a
specific named item/model and for an accessory requested for a named item.

## Specificity

Apply the first matching level in this precedence order:

1. `exact_model_or_identifier`: Contains an exact model, generation, part
   number, SKU-like identifier, or named device/model that constrains the
   requested item. This includes accessories for a named model, such as
   “Pixel 3 screen protector.”
2. `named_entity_or_title`: Names a brand/store only, person/author, media
   title, proprietary product line, or distinct named item, but no exact model
   identifier. A brand plus a generic product type does not use this label;
   classify it at level 3.
3. `product_type_with_constraints`: Names a product type plus brand, color,
   size, material, audience, compatibility target without an exact model,
   quantity, feature, or other explicit constraint.
4. `product_type_only`: Names only a generic product type/category, such as
   “paper cups,” “VR goggles,” or “luggage.”
5. `broad_need_or_occasion`: Expresses an occasion, recipient, activity,
   problem, theme, or broad merchandise need without one definite product
   type.
6. `none_or_unclear`: No defensible specificity level can be inferred.

Do not treat an unexplained token as an exact identifier merely because it
contains a number.

## Commerce scope

- `in_scope`
- `out_of_scope`
- `ambiguous`

## Required output

```json
{
  "item_id": "intent_0001",
  "goal": "one goal label",
  "object": "one object label",
  "specificity": "one v2 specificity label",
  "commerce_scope": "in_scope|out_of_scope|ambiguous",
  "confidence": "low|medium|high",
  "rationale": "one concise sentence"
}
```

The two judges work independently, use low effort, receive no weak labels, and
return structured JSON only.
